# machine-learning-notebooks

AI研修のハンズオン用リポジトリです。  
スライド: https://speakerdeck.com/mixi_engineers

## 注意事項
このリポジトリは`hands-on`ディレクトリの各チャプター内にある`*.ipynb`ファイルに対して、GCPのJupyterLab環境内で動作させることを想定しています。  
Vertex AIやGCS等を使用している部分は、研修以外の環境でやる場合、適宜変更する必要があるので注意してください。  
また、04のハンズオンは社外秘のデータセットのため、研修以外では動作しません。

## ブランチについて
ブランチは、`master`と`solutions`に分かれています。  
ブランチを`solutions`に切り替えると、各ハンズオンの答えがあるので、どうしても困った時は参考にしてください。

## ハンズオン目次
- 00_intro_jupyter_notebook
- 01_image_classification
  - 01ex_pruning
- 02_transfer_learning
- 03_deploy_and_serving
  - 03ex_parameter_tuning
- 04_predict_structured_data  (データセットが社外秘のため研修での利用限定)
