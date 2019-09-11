from data_loader.data_gen import *
from models.unet import *
from argparse import ArgumentParser
from evaluate import *
import pandas as pd
def thresh(img):
    img[img>0.5]=1
    img[img<=0.5]=0
    return img
def save_results(results,save_path):
    for i,img in enumerate(results):
        img =img[:,:,0]
        img = thresh(img)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
def test(args):
    data_gen = Data_generator()
    size = int(args['size'])
    target_size = (size,size)
    test_data,masks = data_gen.test_gen(target_size=target_size)
    if args['model']=='unet':
        unet = Unet(input_size=target_size)
        unet.load('unet')
    elif args['model']=='attention':
        unet = Attention_unet(input_size=target_size)
        unet.load('attention_unet')
    elif args['model']=='dense':
        unet = Dense_unet(input_size=target_size)
        unet.load('dense')
    results = unet.predict(test_data)
    path = 'data/results/'+args['model']+'_results'
    save_results(results,path)
    result = np.zeros((len(test_data),6))
    for i,img in enumerate(test_data):
        img=img[:,:,0]
        r =results[i][:,:,0]
        r = thresh(r)
        r = r.astype(np.int32)
        vis_segmentation(img, r,target_size,args['model'],i)
        result[i]= cal_accuracy(masks[i],r)
    df = pd.DataFrame(result,columns=['TP','TN','FP','FN','ACC','DICE'])
    df.to_csv('data/results/'+args['model']+'_result.csv')



if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',choices=['unet','attention','dense'],default='attention')
    parser.add_argument('--size',default=256)
    args=vars(parser.parse_args())
    test(args)
