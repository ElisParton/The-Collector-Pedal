# This is the beginnings of a guitar pedal (The collector) designed to learn the sound of other guitar pedals.
I will refer to the guitar pedal that is doing the learning as "The collector" and the guitar pedal that is being trained on as "The subject"

## How it works
A training dataset is played from the collector into the subject. The altered sound coming back from the subject is then recorded back into the collector. A neural network is then trained on this data to learn the patterns of what happens within the subject. After training, the model is used for inference to emulate the pedal.

## What I've done so far
The c file written so far cover a basic implementation of a general neural network with the important caveat that dynamic memory allocations such as malloc and free are not used anywhere. The entire library relies on fixed size buffers and stack allocation to allow the easy eventual implementation to stm32.

## Plans
After finishing the creation of the pedal for general-purpose computing I will work towards hardware design and embedded system implementation.