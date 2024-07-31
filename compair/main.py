
from pydub import AudioSegment 
  
# Open an mp3 file 
song = AudioSegment.from_file("compair/pa_pa_pa.mp3", 
                              format="mp3") 
  
# pydub does things in milliseconds 
ten_seconds = 10 * 1000
print(type(ten_seconds))  
# song clip of 10 seconds from starting 
first_10_seconds = song[:ten_seconds] 
  
# save file 
first_10_seconds.export("compair/second.mp3", 
                        format="mp3") 
print("New Audio file is created and saved") 


