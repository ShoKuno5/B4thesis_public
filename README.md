# graduation-report
## 構成
構成は以下のようになっています．
<pre>
.
├── chapters
│   ├── appendix.tex
│   ├── intro.tex
│   ├── method-3body.tex
│   ├── method-percolation.tex
│   ├── notation.tex
│   ├── prev.tex
│   └── summary.tex
├── images
│   ├── hoge.pdf (略)
│   └── fuga.pdf (略)
├── 03200613_加藤雅己_最終版.pdf
├── main.tex
├── reference.bib
├── skeleton_system_B_Japanese
├── systemB.cls
└── README.md
</pre>
## 説明
main.tex に ./chapter 内の tex ファイルを subfile する分割コンパイルをしていて，main.tex で reference.bib と systemB.cls を読み込んでいます．
画像ファイル（pdf）は ./images 以下にあります．
