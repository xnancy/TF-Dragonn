# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code. Basic workflow:

0. tf/celltype name -> raw peaks files and signal files (using our database, for internal use only)
    * Not yet implemented
1. raw peaks files -> regions & labels
    * Implemented in intervals.py
2. regions + signal files -> memmapped data w streaming (large scale data) or data arrays in memory (small scale data)
    * in memory option will be implemented using genomedatalayer
    * memmapped/streaming option will be implemented later using genomeflow
3. data + labels -> trained model
    * preliminary implementation in models.py using keras
    * large scale implementation will rely on tensorflow
4. postprocessing (bigwigs with scoress, motif clustering)
    * preliminary implementation in notebook
    * for tensorflow models, either port deeplift to tf or port tf models to keras 
