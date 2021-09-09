import os

location = '/home/tim/Documents/Datasets/hu_dysarthria_final_data/final_data'

speakers = os.listdir(location)

os.system('ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 0.1 -q:a 9 -acodec libmp3lame 100msSilence.mp3') #('ffmpeg -filter_complex aevalsrc=0 -t 0.1 -ar 16000 100msSilence.mp3')

for speaker in speakers:
    loc = os.path.join(location, speaker)
    if os.path.isdir(loc) and speaker[0] == 'M' or speaker[0] == 'F':
        loc = os.path.join(loc, 'audio_1')
        if os.path.isdir(loc):
            audio_files = os.listdir(loc)
            for af in audio_files:
                if af.endswith('.mp3'):
                    a = os.path.join(loc, af)
                    a_copy = os.path.join(loc, 'copy_' + af)
                    command = '\"concat:100msSilence.mp3|{}|100msSilence.mp3\"'.format(a)
                    #print(command)
                    os.system('ffmpeg -i {} -c copy {}'.format(command, a_copy))
