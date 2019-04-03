sudo apt-get install mpg123
mpg123 -w output.wav input.mp3

or

 for f in ls; do ffmpeg -i $f $(basename $f).wav

directlory configs:
	 
Base Dir: /Books/chunks/1/xxx.wav
	  /Books/chunks/2/yyy.wav
	  /Books/texts/1/xxx.txt
	  /Books/texts/2/yyy.wav

getsentence.py
solvewithgoogle.py
