# Specification

## Signal Recorder

- [ ] いろんな受信機が考えられるので、Signal Recorderは差し替え可能とする。
- [ ] データベースに、Recorderの種類（SoftRock 9.0 Lite, RTL-SDR + Spyverter）などを記録する。
- [ ] 日単位でディレクトリを分ける。（TBD: UTC?  ローカルタイム?）
- [ ] -1秒から9秒まで、10秒毎にファイルを分ける。
- [ ] ファイルは.wavフォーマットとする。
- [ ] クリーナー（Signal Files Cleaner）も作る。

## Characteristics Extractor

特徴抽出器

## Bayesian Inference

ベイズ推定器

## Signal Files Cleaner

- [ ] 定期的に音声ファイルを削除する。
- [ ] オプションで、可逆圧縮形式でアーカイブするようにする。

## Miscellaneous

### Configuration

- [ ] configファイルを作り、ユーザー設定値を書き込む。

### Database

- [ ] スキーマはどうしよう。あまり凝ったものにせず、必要最小限のシンプルなものから始める。

   - [ ] 日時（TBD: UTC?  ローカルタイム?）
   - [ ] 時刻オフセット（単位はミリ秒。-1000ならば、1秒前から受信した音声）
   - [ ] 受信周波数（kHz単位まで。BFO+サイドトーン周波数）
   - [ ] CW復調サイドトーン周波数（+のときはCW Reverse。つまり、BFOが対象周波数より下にある）
   - [ ] Signal Recorderの種別
   - [ ] 抽出した特徴パラメタ
   - [ ] 判定結果★TBD

> CW Reverse: http://www.kb6nu.com/cw-bass-ackwards/

### Beacon Stations

- [ ] ビーコン局の情報をどうするか。ビーコン局の場所やコールサインは変わるかも知れない。
- [ ] SQLiteに持たせるのは大袈裟だし、メンテが大変そうなので、もっと簡単なテキスト形式ファイルにしたい。YAMLかJSONにするか。。。
   - スロット番号（タイムフレームと周波数）単位に枝を切って、その中に
      - 運用開始年月日
	  - コールサイン
	  - 位置
     を書くようにするか。

### Unit Tests

ディレクトリtestで、python test_all.pyを実行すると単体テストができる。
