# Anomaly detection

#### List of issues:
* ~~Transformation is too long~~ -> Possibly without solution<br>
* ~~Training is too long~~ -> Moved to CPU with GPU<br>
* ~~Training loss is too big~~ -> Solved by model tuning and changes<br>
* ~~Model is not robust to camera position change~~ -> Dependant on data<br>
* ~~Validation Loop for training and inference takes too long~~ -> Create snapshots with selected anomalies and infer them saparatelly<br>
* Validation and inference part needs to be reworked

### TODO-MUST:
 * ~~Parse video to open pose vectors~~ <br>
 * ~~Transform open pose vectors to numpy~~<br>
 * ~~Prepare model and traing process~~<br>
 *   Infer model with apropriate cost function<br>
 *  Combine result with output video for presentation<br>

 ### TODO-Good to do:
 * Get more data<br>
 * Prepare process how to train and improve model iterativelly<br>
 * Prepare POC of validation set <br>
 * Create POC for model combination with different models<br>
 
### Questions
* Would it be good to use distance to all points instead of coordinates?<br>
* Is there possibility to combine different models for better results like in inception networks?<br>
* How much overhead will OpenPose mean, how much should we optimize inference?<br>

 
 
