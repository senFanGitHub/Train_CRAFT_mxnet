from core import Train_Char_Det


def Do_train():
    Det =Train_Char_Det(Synth_gt=None,
                        ReCTS_root="/data3/fansen/CVProject/text-detection/DataSets/ReCTs/img",
                        pretrained ='Models/ReCTS.params')
    Det.training()

if __name__ == "__main__":
    
    Do_train()
