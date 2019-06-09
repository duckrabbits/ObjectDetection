# ObjectDetection

Kerasを利用したFaster R-CNN実装により物体検出を行います。

## クックスタート

Kaggleのシンプソンズデータセットで学習・推定を行う手順を説明します。

[Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

### データセットの用意

- Kaggleからダウンロードしたthe-simpsons-characters-dataset.zipを解凍します。
- simpsons_dataset.zipを解凍します。
- 解凍したsimpsons_datasetフォルダをプロジェクトフォルダ配下に格納します。
- annotation.txtもプロジェクトフォルダ配下に格納します。
 （kaggleからダウンロードしたannotation.txtではなく、ここにあるannotation.txtを使いましょう）

### 学習

以下のコマンドを実行してください。

`python train.py -p annotation.txt`

以下のように設定ファイルのパスが出力されるので記録してください。推定時に使います。

"""
path to config file : ./save/train_20190309-220050_config.pickle
"""

エポック数なども引数で変更できます。詳細はtrain.pyを参照してください。

### 推定

以下のコマンドを実行してください。

`python predict.py -i [推定したい画像を入れたディレクトリのパス] -c [学習時に保存された設定ファイルのパス]`
