# Computer Vision by using ML with Spark.

SparkUI will be available at [http://192.168.0.110:4040/jobs/](http://192.168.0.110:4040/jobs/)

If we uses Dataset.checkpoint(), it returns a checkpointed version of this dataset. 
Checkpointing can be used to truncate the logical plan of this dataset, which is especially useful 
in iterative algorithms where the plan may grow exponentially. 
It will be saved to files inside the checkpoint directory set with SparkContext#setCheckpointDir.
But using javaSparkContext.parallelize(labeledImages).cache() can be significantly faster in order to use 
caching into memory.

Structure of DataSet
![**Structure of DataSet**](https://raw.githubusercontent.com/Evegen55/car_number_recognizer/master/src/test/resources/for_readme/dataset_structure.PNG)



## License

One of a logic branch here grows from [DigitRecognizer](https://github.com/klevis/DigitRecognizer) project by [Klevis Ramo](https://github.com/klevis).

Thank you very much for starting point!

Copyright (C) 2017 - 2017 Evgenii Lartcev (https://github.com/Evegen55/) and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.