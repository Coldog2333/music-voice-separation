# music-voice-separation

## Introduction

We design a multi-layers LSTM (DLSTM) for music voice separation. It is a task that given a signal mixture, generate the music and human voice separately. Firstly extract the magnitude of the mixture, and feed it into DLSTM and generate the magnitude of the music and voice. And finally generate the music and human voice from the mixture.

## Python Version
- **3.6**

## Modules needed
- **torch**
- **scipy**
- **pyaudio**
- **wave**
- **matplotlib**
- **psutil**

## Contact

- jiangjf6@mail2.sysu.edu.cn
- caixy3@mail2.sysu.edu.cn