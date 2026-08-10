from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math


with open('/media/pbclab/Elements/qpp/BERTQPP/pklfiles/msmarco_train_mrr_10.pkl', 'rb') as f:
    q_map_dic_train=pickle.load(f)


train_set=[]

for key in q_map_dic_train:
    # try:
        qtext=q_map_dic_train[key]["qtext"]
        firstdoctext=q_map_dic_train[key]["doc_text"]
        actual_map = q_map_dic_train[key] ["performance"]
        train_set.append( InputExample(texts=[qtext,firstdoctext],label=actual_map ))
    # except:
    #     continue


batch_size=16
epoch_num=1
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1) #10% of train data for warm-up
model_name='bert-base-uncased'

model = CrossEncoder(model_name, num_labels=1)
model_name="models/msmarco_tuned_model-ce_"+model_name+"_e"+str(epoch_num)+'_b_mrr10'+str(batch_size)+'_ep1'
# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=epoch_num,
          warmup_steps=warmup_steps,
          output_path=model_name)
model.save(model_name)
