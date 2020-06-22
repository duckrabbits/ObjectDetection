
書き換えるところ

K.image_dim_ordering  -->  K.image_data_format

'tf' --> 'channels_last'

'th' --> 'channels_first'


train.pyとpredict.pyはObjectdetectionフォルダ直下へアップロードして差し替える

それ以外のpyファイルはmodelフォルダ直下へアップロードして差し替える

https://keras.io/backend/

https://github.com/keras-team/keras/issues/12649