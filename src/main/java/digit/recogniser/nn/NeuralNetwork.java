package digit.recogniser.nn;

import digit.recogniser.data.IdxReader;
import digit.recogniser.data.LabeledImage;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

import static digit.recogniser.data.IdxReader.VECTOR_DIMENSION;

public class NeuralNetwork {

    private final static Logger LOGGER = LoggerFactory.getLogger(NeuralNetwork.class);

    private SparkSession sparkSession;
    private MultilayerPerceptronClassificationModel model;

    private static final String PATH_TO_TRAINED_SET = "TrainedModels";
    private static final String FOLDER_ROOT = "/ModelWith";
    private static final String PATH_TO_TRAINED_SET_INIT = PATH_TO_TRAINED_SET + FOLDER_ROOT;

    public static final int NEURAL_OUTPUT_CLASSES = 10;

    private boolean isModelUploaded = false;

    public void init(final int initialTrainSize, final boolean erasePreviousModel) {
        initSparkSession();
        if (model == null || erasePreviousModel) {
            try {
                LOGGER.info("Load model from trained set: " + FOLDER_ROOT + initialTrainSize);
                model = MultilayerPerceptronClassificationModel.load(PATH_TO_TRAINED_SET_INIT + initialTrainSize);
                isModelUploaded = true;
            } catch (Exception e) {
                /*
                It tries to load metadata firstly
                so it could throw next exception:
                org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:<your path>/TrainedModels/ModelWith30000/metadata
                 */
                if (e.getClass().getName().equals("org.apache.hadoop.mapred.InvalidInputException")) {
                    LOGGER.error("The model doesn't exist");
                } else {
                    e.printStackTrace();
                }
            }
        }
    }

    public void train(Integer trainData, Integer testFieldValue, final boolean saveOrNot, int[] layers) {
        initSparkSession();

//        Dataset<Row> df1 = spark.read()
//                .format("csv").option("inferSchema", "true")
//                .option("header", "false")
//                .load(filename);

        // TODO: 27.06.18 try cache
        List<LabeledImage> labeledImages = IdxReader.loadData(trainData);
        List<LabeledImage> testLabeledImages = IdxReader.loadTestData(testFieldValue);
//        System.gc();
        Dataset<Row> train = sparkSession.createDataFrame(labeledImages, LabeledImage.class).cache();
        Dataset<Row> test = sparkSession.createDataFrame(testLabeledImages, LabeledImage.class).cache();
        labeledImages = null;
        testLabeledImages = null;
        System.gc();

        if (layers == null) {
            //DEFAULT VALUE
            //first layer is an image 28x28 pixels -> 784 pixels
            //last layer is a digit from 0 to 9, the output is a one dimensional vector of size 10.
            //The values of output vector are probabilities that the input is likely to be one of those digits.
            layers = new int[]{VECTOR_DIMENSION, 128, 64, NEURAL_OUTPUT_CLASSES};
        }

        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier("My MultilayerPerceptronClassifier")
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);

        model = trainer.fit(train);

        if (saveOrNot) {
            // after saving we have to load NN from trained data set
            try {
                model.save(PATH_TO_TRAINED_SET + FOLDER_ROOT + trainData);
            } catch (IOException e) {
                LOGGER.error("Smth went wrong" + e);
                e.printStackTrace();
            }
            init(trainData, true);
            if (isModelUploaded) {
                LOGGER.info("NEURAL NETWORK trained with " + trainData + " has been uploaded successfully");
            }
        }
        evalOnTest(test);
        evalOnTest(train);

    }

    private void evalOnTest(final Dataset<Row> rowDataset) {
        final Dataset<Row> result = model.transform(rowDataset);
        final Dataset<Row> predictionAndLabels = result.select("prediction", "label");
        final MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");

        LOGGER.info("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
    }

    private void initSparkSession() {
        if (sparkSession == null) {
            sparkSession = SparkSession.builder()
                    .master("local[*]")
                    .appName("Car's number recognizer")
                    .getOrCreate();
        }
//        sparkSession.sparkContext().setCheckpointDir("checkPoint");
    }

    /**
     * the output labeled image consists of vector with features BEFORE prediction
     * and
     * label AFTER prediction
     *
     * @param labeledImage
     * @return
     */
    public LabeledImage predict(LabeledImage labeledImage) {
        double predict = model.predict(labeledImage.getFeatures());
        labeledImage.setLabel(predict);
        return labeledImage;
    }

    public boolean isModelUploaded() {
        return isModelUploaded;
    }
}
