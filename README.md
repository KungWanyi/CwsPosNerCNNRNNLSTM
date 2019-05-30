# CwsPosNerEntityRecognition
Chinese &amp; English Cws Pos Ner Entity Recognition implement using CNN bi-directional lstm and crf model with char embedding.基于字向量的CNN池化双向BiLSTM与CRF模型的网络，可能一体化的完成中文和英文分词，词性标注，实体识别。主要包括原始文本数据，数据转换,训练脚本,预训练模型,可用于序列标注研究.注意：唯一需要实现的逻辑是将用户数据转化为序列模型。分词准确率约为93%，词性标注准确率约为90%，实体标注（在本样本上）约为85%。
