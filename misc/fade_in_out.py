import os
# import pydub

# Thought: I don't think we should fade in, as it will decrease the volume of the spoken language in the beginning of the clip

location = '/home/tim/Documents/Datasets/hu_dysarthria_final_data/final_data'

speakers = os.listdir(location)

os.system('ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 0.1 -q:a 9 -acodec libmp3lame 100msSilence.mp3') #('ffmpeg -filter_complex aevalsrc=0 -t 0.1 -ar 16000 100msSilence.mp3')

for speaker in speakers:
    loc = os.path.join(location, speaker)
    if os.path.isdir(loc) and speaker[0] == 'M' or speaker[0] == 'F':
        l = os.listdir(loc)
        print(l)
        for lo in l:
            lo = os.path.join(loc, lo)
            print(lo)
            if os.path.isdir(lo):
                audio_files = os.listdir(lo)
                for af in audio_files:
                    if af.endswith('.mp3'):
                        a = os.path.join(lo, af)
                        a_copy = os.path.join(lo, 'copy_' + af)
                        command = '\"concat:100msSilence.mp3|{}|100msSilence.mp3\"'.format(a)
                        #print(command)
                        os.system('ffmpeg -i {} -c copy {}'.format(command, a_copy))
