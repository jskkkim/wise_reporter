import flashBased as fb
import connect2ES as es
import math

#1. connect to elasticsearch
getDoc = es.ScouterHandler() #__init__ 실행#contents 한줄
#qbody = es.ScouterHandler.make_keyword_query_body("경찰",("news_id","extContent","analyzed_text.sentence.WSD.text","analyzed_text.sentence.WSD.type","category","analyzed_title.sentence.WSD.text","analyzed_title.sentence.WSD.type"))
qbody = es.ScouterHandler.make_all_query_body(".*다.*",("news_id","extContent","analyzed_text.sentence.WSD.text","analyzed_text.sentence.WSD.type","category","analyzed_title.sentence.WSD.text","analyzed_title.sentence.WSD.type"))
#qbody = es.ScouterHandler.make_all_query_body("유튜브",("news_id","extContent","analyzed_text.sentence.WSD.text","analyzed_text.sentence.WSD.type","category","analyzed_title.sentence.WSD.text","analyzed_title.sentence.WSD.type"))
docs = getDoc.search(qbody,'test_newspaper') # querybody, keyword

#flash = fb.FlashBased((int)(math.sqrt(len(docs)/2)/2),0.95,docs) #k-value, similarity rate, documents(by elasticsearch)

flash = fb.FlashBased(112,0.95,docs)

ftime,fb_result = flash.runFlash() #return elsped Time, duplicated documents set
deleteIt = flash.eliminateDup() # deleteIt: 지워야 할 doc id list

print("delete: " + str(len(deleteIt)))
