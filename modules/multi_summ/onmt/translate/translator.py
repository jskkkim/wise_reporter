#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import codecs
import os
import math
import time

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
from onmt.translate.beam_search import BeamSearch
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.utils.misc import set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores
import pickle


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(
        self,
        model,
        fields,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_score=True,
        logger=None
    ):

        self.model = model
        self.fields = fields
        tgt_field = self.fields["tgt"][0][1].base_field
        tgt_field.eos_token = '</s>'
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)
        self.opt=opt
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length

        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

        self.beam_size = opt.beam_size
        self.random_sampling_temp = opt.random_sampling_temp
        self.sample_from_topk = opt.random_sampling_topk

        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        self.tgt_reader = inputters.str2reader["text"].from_opt(opt)
        self.replace_unk = opt.replace_unk
        self.data_type = opt.data_type
        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.report_time = opt.report_time
        self.fast = opt.fast

        self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(opt.seed, self.cuda)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def translate(
        self,
        src,
        tgt=None,
        src_dir=None,
        batch_size=None,
        attn_debug=False,
        data_iter=None
    ):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            tgt_path (str): filepath of target data or None
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        data = inputters.build_dataset(
            self.fields,
            self.data_type,
            src=src,
            src_reader=self.src_reader,
            tgt=tgt,
            tgt_reader=self.tgt_reader,
            src_dir=src_dir,
            use_filter_pred=self.use_filter_pred, bert=self.opt.bert, morph=self.opt.korean_morphs
        )

        cur_device = "cuda" if self.cuda else "cpu"

        # data_iter = inputters.OrderedIterator(
        #     dataset=data,
        #     device=cur_device,
        #     batch_size=batch_size,
        #     train=False,
        #     sort=False,
        #     sort_within_batch=True,
        #     shuffle=False
        # )
        builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch_ in data_iter:
            batch = bbbb(batch_)
            # batch_data = self.translate_batch(
            #     batch, data.src_vocabs, attn_debug, fast=self.fast
            # )
            batch_data = self.translate_batch(
                batch, batch.dataset.src_vocabs, attn_debug, fast=self.fast
            )
            # batch_data = self.translate_batch(
            #     batch, data.src_vocabs, attn_debug, fast=self.fast
            # )
            translations = builder.from_batch(batch_data)
            return translations
            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))
                        print(list(trans.attns[0].max(1)[1].cpu().detach().numpy()))
                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                self._log(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt)
                    self._log(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt)
                    self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                pred_words_total / total_time))

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def sample_with_temperature(self, logits, sampling_temp, keep_topk):
        if sampling_temp == 0.0 or keep_topk == 1:
            # For temp=0.0, take the argmax to avoid divide-by-zero errors.
            # keep_topk=1 is also equivalent to argmax.
            topk_scores, topk_ids = logits.topk(1, dim=-1)
        else:
            logits = torch.div(logits, sampling_temp)

            if keep_topk > 0:
                top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
                kth_best = top_values[:, -1].view([-1, 1])
                kth_best = kth_best.repeat([1, logits.shape[1]]).float()

                # Set all logits that are not in the top-k to -1000.
                # This puts the probabilities close to 0.
                keep = torch.ge(logits, kth_best).float()
                logits = (keep * logits) + ((1-keep) * -10000)

            dist = torch.distributions.Multinomial(
                logits=logits, total_count=1)
            topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
            topk_scores = logits.gather(dim=1, index=topk_ids)
        return topk_ids, topk_scores

    def _translate_random_sampling(
        self,
        batch,
        src_vocabs,
        max_length,
        min_length=0,
        sampling_temp=1.0,
        keep_topk=-1,
        return_attention=False
    ):
        """Alternative to beam search. Do random sampling at each step."""

        assert self.beam_size == 1

        # TODO: support these blacklisted features.
        assert self.block_ngram_repeat == 0

        batch_size = batch.batch_size

        end_token = self._tgt_eos_idx

        # Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        use_src_map = self.copy_attn

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                src_vocabs,
                batch.src_map if use_src_map else None
            )
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            results["gold_score"] = [0] * batch_size

        memory_lengths = src_lengths
        src_map = batch.src_map if use_src_map else None

        if isinstance(memory_bank, tuple):
            mb_device = memory_bank[0].device
        else:
            mb_device = memory_bank.device

        # seq_so_far contains chosen tokens; on each step, dim 1 grows by one.
        seq_so_far = torch.full(
            [batch_size, 1], self._tgt_bos_idx,
            dtype=torch.long, device=mb_device)
        alive_attn = None

        for step in range(max_length):
            decoder_input = seq_so_far[:, -1].view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=torch.arange(batch_size, dtype=torch.long)
            )

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Note that what this code calls log_probs are actually logits.
            topk_ids, topk_scores = self.sample_with_temperature(
                    log_probs, sampling_temp, keep_topk)

            # Append last prediction.
            seq_so_far = torch.cat([seq_so_far, topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

        predictions = seq_so_far.view(-1, 1, seq_so_far.size(-1))
        attention = (
            alive_attn.view(
                alive_attn.size(0), -1, 1, alive_attn.size(-1))
            if alive_attn is not None else None)

        for i in range(topk_scores.size(0)):
            # Store finished hypotheses for this batch. Unlike in beam search,
            # there will only ever be 1 hypothesis per example.
            score = topk_scores[i, 0]
            pred = predictions[i, 0, 1:]  # Ignore start_token.
            m_len = memory_lengths[i]
            attn = attention[:, i, 0, :m_len] if attention is not None else []

            results["scores"][i].append(score)
            results["predictions"][i].append(pred)
            results["attention"][i].append(attn)

        return results

    def translate_batch(self, batch, src_vocabs, attn_debug, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)
        """
        with torch.no_grad():
            if self.beam_size == 1:
                return self._translate_random_sampling(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    return_attention=attn_debug or self.replace_unk)
            if fast:
                return self._fast_translate_batch(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(batch, src_vocabs)

    def _run_encoder(self, batch):
        cls_bank=None
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)
        if self.opt.bert:
            mask = (src[:, :, 0].transpose(0, 1).data.eq(0) ^ torch.ones_like(src.transpose(0, 1).squeeze(2),
                                                                              dtype=torch.bool))
            enc_states, memory_bank, _ = self.model.encoder(src.squeeze(2).transpose(0, 1), attention_mask=mask,
                                                     output_all_encoded_layers=False, output_embeddings=True, adapter=True)
            # enc_states = enc_states.transpose(0, 1)
            # memory_bank = memory_bank.transpose(0, 1)
            # cls_bank = memory_bank[:, 0:1, :]
            memory_bank = torch.cat(
                [memory_bank.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            enc_states = torch.cat([enc_states.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))],
                                  0)
            # enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths, adapter=True)
            # memory_bank = torch.cat([memory_bank[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            # enc_states = torch.cat([enc_states[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            # print()
        elif self.opt.korean_morphs:
            mask = (src[:, :, 0].transpose(0, 1).data.eq(0) ^ torch.ones_like(src.transpose(0, 1).squeeze(2),
                                                                              dtype=torch.uint8))
            enc_states, memory_bank, _ = self.model.encoder(src.squeeze(2).transpose(0, 1), attention_mask=mask,
                                                            output_all_encoded_layers=False, output_embeddings=True,
                                                            adapter=True)
            # enc_states = enc_states.transpose(0, 1)
            # memory_bank = memory_bank.transpose(0, 1)

            memory_bank = torch.cat(
                [memory_bank.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            enc_states = torch.cat(
                [enc_states.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))],
                0)
            # enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths, adapter=True)
            # memory_bank = torch.cat([memory_bank[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            # enc_states = torch.cat([enc_states[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
        elif self.opt.segment:
            aa = []
            for a in src.squeeze(2).transpose(0, 1):
                bb = []
                ma = True
                for b in a:
                    if self.model.segment[int(b)]:
                        bb.append(ma)
                        ma = not ma
                    else:
                        bb.append(ma)
                aa.append(bb)
            mask = (src[:, :, 0].transpose(0, 1).data.eq(0) ^ torch.ones_like(src.transpose(0, 1).squeeze(2),
                                                                              dtype=torch.bool))
            enc_states, memory_bank, _ = self.model.encoder(src.squeeze(2).transpose(0, 1),
                                                            token_type_ids=torch.tensor(aa).type(torch.int64).to(
                                                                'cuda'),
                                                            attention_mask=mask, output_all_encoded_layers=False,
                                                            output_embeddings=True, adapter=True)
            # enc_states = enc_states.transpose(0, 1)
            # memory_bank = memory_bank.transpose(0, 1)

            memory_bank = torch.cat(
                [memory_bank.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            enc_states = torch.cat(
                [enc_states.transpose(0, 1)[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))],
                0)
            # enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths, adapter=True)
            # memory_bank = torch.cat([memory_bank[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
            # enc_states = torch.cat([enc_states[:src_lengths[i], i:i + 1, :] for i in range(src_lengths.size(0))], 0)
        else:
            enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))

        return src, enc_states, memory_bank, src_lengths, cls_bank

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        src_vocabs,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None,
        cls_bank=None
    ):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

	# Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch

        # cls_bank = None
        # cls_bank = memory_bank[:, 0:1, :]

        # dec_out, dec_attn = self.model.decoder(
        #     decoder_in, memory_bank, memory_lengths=memory_lengths, step=step, adapter=True, cls_bank=cls_bank)
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step, adapter=True)

        # Generator forward.
        if not self.copy_attn:
            attn = dec_attn["std"]
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        return log_probs, attn

    def _fast_translate_batch(
        self,
        batch,
        src_vocabs,
        max_length,
        min_length=0,
        n_best=1,
        return_attention=False
    ):
        # TODO: support these blacklisted features.
        assert not self.dump_beam
        assert self.global_scorer.beta == 0

        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {}
        results["predictions"] = None
        results["scores"] = None
        results["attention"] = None
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                src_vocabs,
                batch.src_map if use_src_map else None
            )
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            results["gold_score"] = [0] * batch_size

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=beam.batch_offset
            )

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results

    def _translate_batch(self, batch, src_vocabs):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.

        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size
        # results = {}
        # results["predictions"] = [[[0]]] * batch_size
        # results["scores"] = [[0]] * batch_size
        # results["attention"] = [[torch.zeros(1, 400)]] * batch_size
        # results["batch"] = batch
        # results["gold_score"] = [0] * batch_size
        # if batch.indices[0] < 570:
        #     return results
        beam = [onmt.translate.Beam(
            beam_size,
            n_best=self.n_best,
            cuda=self.cuda,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=self.min_length,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs)
            for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths, cls_bank = self._run_encoder(batch)
        self.model.decoder.init_state(torch.cat([src[:src_lengths[i], i:i + 1, :]
                                                 for i in range(src_lengths.size(0))], 0), memory_bank, enc_states)


        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None, cls_bank=cls_bank)
            self.model.decoder.init_state(torch.cat([src[:src_lengths[i], i:i + 1, :]
                                                 for i in range(src_lengths.size(0))], 0), memory_bank, enc_states)
        else:
            results["gold_score"] = [0] * batch_size
        src_lengths = src_lengths.sum().unsqueeze(0)
        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)
        if cls_bank is not None:
            cls_bank = tile(cls_bank, beam_size, dim=1)
        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.current_predictions for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = self._decode_and_generate(
                inp, memory_bank, batch, src_vocabs,
                memory_lengths=memory_lengths, src_map=src_map, step=i, cls_bank=cls_bank
            )
            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths[j]])
                select_indices_array.append(
                    b.current_origin + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            scores, ks = b.sort_finished(minimum=self.n_best)
            hyps, attn = [], []
            for times, k in ks[:self.n_best]:
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths,
                      src_vocabs, src_map, cls_bank=None):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map, cls_bank=cls_bank)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt_in
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            try:
                msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                    name, score_total / words_total,
                    name, math.exp(-score_total / words_total)))
            except:
                msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                    name, 1,
                    name, 1))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, tgt_path),
            stdin=self.out_file, shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg

class bbbb:
    def __init__(self, batch):
        self.batch_size = 1
        self.src = (batch['src'].to(torch.device('cuda')).unsqueeze(2), batch['src_lengths'].to(torch.device('cuda')))
        self.dataset = lambda: None
        self.dataset.src_vocabs = [lambda: None]
        self.dataset.src_vocabs[0].itos = batch['itos']
        self.dataset.src_vocabs[0].stoi = batch['stoi']
        self.indices = torch.tensor([0], dtype=torch.int64, device=torch.device('cuda'))
        self.src_map = self.makeSrcmap(batch['src_map'], batch['src_lengths']).to(torch.device('cuda'))
        self.batch_size = 1

    def makeSrcmap(self, data, lengths):
        src_size = max([t.size(0) for t in data])
        src_vocab_size = len(self.dataset.src_vocabs[0].itos)
        alignment = torch.zeros(src_size, len(data), src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[j, i, t] = 1
        return alignment

