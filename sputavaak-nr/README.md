### Caveats

Instructions are known to work on an OSX box with brew installs.
Your mileage may vary on other operating systems.
Familiarity with a Terminal / Shell equivalent on your OS of choice
will make it really easy.

### What does this do ?
This is a small wrapper on top of DeepFilterNet that is targeted to
remove noise from audio recordings that are mostly voice (like a podcast).


### Build instructions

1. `git checkout` this repository to your local directory. For eg. `gitprojects/sputavaak`.
If this is unclear Google/ChatGpt should be able to assist
2. `cd gitprojects/sputavaak`: Change directory to this location.
3. Execute  `pipenv sync -d` and then do
```angular2html
$ pipenv run python -m build
$ pipenv run pip3 install dist/*.whl
```

### To run 
1. `pipenv shell` : Start a pipenv shell
2. `sputavaaknr --help` : Gives help instructions.
3. For eg. `sputavaaknr --input noisyaudio.m4a  --output cleaned-audio.mp3  -bitrate 192k`
