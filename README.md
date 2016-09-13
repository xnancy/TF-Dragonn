# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code. Basic workflow:

0. tf/celltype name -> raw peaks files and signal files (using our database, for internal use only)
    * Not yet implemented
1. raw peaks files -> regions & labels
    * Available in intervals.py
    * TODO: regions & scores (for regression)
2. regions + signal files -> memmapped data w streaming (large scale data) or data arrays in memory (small scale data)
    * Prototyped for sequence-only in notebook using genomedatalayer in memory extractor
    * TODO:
        * move prototype code to .py file
	* add bigwig extraction using genomedatalayer
	* add utilties for multi-sample data organization
        * added memmapped/streaming data option using genomedatalayer
	* port to tensorflow using genomeflow
3. data + labels -> trained model
    * SequenceClassifier available in models.py using keras
    * TODO:
        * add sequence+dnase classification model
        * port to tensorflow
4. postprocessing (bigwigs with scoress, motif clustering)
    * Preliminary utilities available in postprocessing_utils.py
    * TODO
        * comparison to known motifs
	* visualization of known motif and clustered grammar hits in browser
	* ranking/prioratization of clustered grammars
        * support tensorflow models by portin deeplift to tf or porting tensorflow models to keras
