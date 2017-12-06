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

public class NeuralNetwork {

    private final static Logger LOGGER = LoggerFactory.getLogger(NeuralNetwork.class);

    private SparkSession sparkSession;
    private MultilayerPerceptronClassificationModel model;

    private static final IdxReader idxReader = new IdxReader();
    private static final String PATH_TO_TRAINED_SET = "TrainedModels";
    private static final String FOLDER_ROOT = "\\ModelWith";
    private static final String PATH_TO_TRAINED_SET_INIT = PATH_TO_TRAINED_SET + FOLDER_ROOT;

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

    public void train(Integer trainData, Integer testFieldValue) throws IOException {
        initSparkSession();

        List<LabeledImage> labeledImages = idxReader.loadData(trainData);
        List<LabeledImage> testLabeledImages = idxReader.loadTestData(testFieldValue);
        Dataset<Row> train = sparkSession.createDataFrame(labeledImages, LabeledImage.class).checkpoint();
        Dataset<Row> test = sparkSession.createDataFrame(testLabeledImages, LabeledImage.class).checkpoint();

        //first layer is an image 28x28 pixels -> 784 pixels
        //last layer is a digit from 0 to 9, the output is a one dimensional vector of size 10.
        //The values of output vector are probabilities that the input is likely to be one of those digits.
        int[] layers = new int[]{784, 128, 64, 10}; // TODO: 12/6/2017 it can gets from UI

        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);

        model = trainer.fit(train);

        // after saving we have to load NN from trained data set
        model.save(PATH_TO_TRAINED_SET + FOLDER_ROOT + trainData);
        init(trainData, true);
        if (isModelUploaded) {
            LOGGER.info("NEURAL NETWORK trained with " + trainData + " has been uploaded successfully");
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
                    .appName("Digit Recognizer")
                    .getOrCreate();
        }

        sparkSession.sparkContext().setCheckpointDir("checkPoint");
    }

    public LabeledImage predict(LabeledImage labeledImage) {
        double predict = model.predict(labeledImage.getFeatures());
        labeledImage.setLabel(predict);
        return labeledImage;
    }

    public boolean isModelUploaded() {
        return isModelUploaded;
    }
}
