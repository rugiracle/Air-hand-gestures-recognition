#  Air handwritten gesture recognition tutorial<br>[work in progress...]
Hand gesture recognition for human computer interaction has numerous applications;
from TV remote control, gaming, Virtual and or augmented reality, in car hand gesture control, etc. 
A successful hand gesture recognition system  requires: 
1. A robust hand detector,
2. A means of gesture spotting to know when a gesture start and ends
3. A gesture recognition method

## Hand trajectory
We will be using mediapipe  https://google.github.io/mediapipe/solutions/hands.html 
to detect and track the hand on a webcam. Mediapipe hand  detector provides us with information of 21 hand landmarks. 
<br>To mark a start of a gesture one opens his hand palm and closes it to mark an end of a gesture one can close the hand<br>
Bellow are samples of air handwritten gestures(hand trajectories)<br>
![alt=air handwritten digits ](./model/AirGesturesSpotted.png "Air handwritten digits")
<br>Although the above gestures are spotted, they still have  noises mostly at their beginnings and ends.
<br> Dropping a few points[one can try a number that works for one's gesturing style ] at the beginning and end, generates neater gestures, as showen in bellow picture<br>
![alt=air handwritten digits ](./model/AirGestures_preprocessedSample.png "Pre-processed Air handwritten digits")
<br>Now that we have digits, we can try to recognize them. 
## Air handwritten digit recognition: Transfer learning

As it is time consuming to record enough samples from various people to capture gesture variations, 
lets first consider transfer learning and utilized a model trained on MNIST digit dataset.<br>
![](./model/Mnist_handwritten_digits.png "handwritten digits")

Use a model trained on MNIST digit dataset to recognize handwritten digits in the air
in front of a webcam. Analyze the results and come up with a more appropriate 
recognition method if necessary.
<br> Air handwritten digits differ from paper handwritten digits due to a number of factors
<br> +  No support, you are writing in the air, producing many gesturing variations even for the same person
<br> +  noisy hand gesture trajectory due to clutter environment 
<br> + Gesture spotting: knowing the exact start and end points of a gesture 
<br>
<br> Now let us train a model capable of recognizing handwritten digits
![](./model/mnist_net.png "handwritten digits")
- [x] Train the model mnist digit dataset<br>
    ![](./model/acc_vs_no_epoch_20_0.55_128.png "Accuracy vs epoch")   
  
- [x] Test the trained model on handwritten digits<br>
   ![](./model/Mnist_confusion_matrix.png "Confusion matrix")
  here a confusion matrix is used as accuracy measure as it provides information on interclass confusion
  
- [x] Test the trained model on air handwritten digits<br>
  ![](./model/AirGestures_predictions.png "air hand written recognition")
  does not look good!
- [x] Online air handwritten testing <br>
  with a bit of patience, one can get his gesture correctly recognized
- [ ] Fine-tuning: Collect enough samples of air handwritten digits for fine-tuning the mnist model<br>
  1. need to collect samples
- [ ] Consider a different  method for online air handwritten digit <br>
