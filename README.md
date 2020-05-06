# Anomaly detection

#### List of issues:
* ~~Transformation is too long~~ -> Possibly without solution<br>
* ~~Training is too long~~ -> Moved to CPU with GPU<br>
* ~~Training loss is too big~~ -> Solved by model tuning and changes<br>
* ~~Model is not robust to camera position change~~ -> Dependant on data<br>
* ~~Validation Loop for training and inference takes too long~~ -> Create snapshots with selected anomalies and infer them saparatelly<br>
* ~~Validation and inference part needs to be reworked~~
* ~~Fine tuning of parameters, fixing-up work so it takes less gpu power~~

### TODO-MUST:
 * ~~Parse video to open pose vectors~~ <br>
 * ~~Transform open pose vectors to numpy~~<br>
 * ~~Prepare model and traing process~~<br>
 * ~~Infer model with apropriate cost function~~<br>
 * ~~Combine result with output video for presentation~~<br>

 ### TODO-Good to do:
 * Get more data<br> -> In progress
 * Prepare process how to train and improve model iterativelly<br> -> Attention based system is in progress
 * Prepare POC of validation set  -> Usable for bigger project with more videos<br>
 * Create POC for model combination with different models -> Attention based model <br>
 * Overtraining?-> Good to do, model is overtrained <br> 
 * False positives evaluator? -> Can be done by annotations <br>
 * Find or create extendable false positives architecture -> Doable using another atention based connection of another network<br>
 * ~~Create POC for model combination with different models~~ <br>
 
### Questions
* ~~Would it be good to use distance to all points instead of coordinates?~~ Nope, degree based system will work better<br>

* ~~Is there possibility to combine different models for better results like in inception networks?~~ Nope, attention and combination of different models will work as the best option 
<br>

 
 
