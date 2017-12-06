package digit.recogniser.nn;

/**
 * Created by klevis.ramo on 11/27/2017.
 */

import digit.recogniser.data.IdxReader;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import digit.recogniser.data.LabeledImage;

import java.io.IOException;
import java.util.List;

public class NeuralNetwork {

    private SparkSession sparkSession;

    private static final IdxReader idxReader = new IdxReader();
    private MultilayerPerceptronClassificationModel model;

    private static final String PATH_TO_TRAINED_SET = "TrainedModels";
    private static final String FOLDER_ROOT = "\\ModelWith";
    private static final String PATH_TO_TRAINED_SET_INIT = PATH_TO_TRAINED_SET + FOLDER_ROOT;
    private boolean isModelUploaded = false;

    public void init(final int initialTrainSize, final boolean erasePreviousModel) {
        initSparkSession();
        if (model == null || erasePreviousModel) {
            try {
                System.out.println("Load model from trained set: " + FOLDER_ROOT + initialTrainSize);
                model = MultilayerPerceptronClassificationModel.load(PATH_TO_TRAINED_SET_INIT + initialTrainSize);
                isModelUploaded = true;
            } catch (Exception e) {
                /*
                It tries to load metadata firstly
                so it could throw next exception:
                org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:<your path>/TrainedModels/ModelWith30000/metadata
                 */
                if (e.getClass().getName().equals("org.apache.hadoop.mapred.InvalidInputException")) {
                    System.out.println("The model doesn't exist");
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

        int[] layers = new int[]{784, 128, 64, 10};

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
            System.out.println("NEURAL NETWORK trained with " + trainData + " has been uploaded successfully");
        }

        evalOnTest(test);
        evalOnTest(train);
    }

    private void evalOnTest(Dataset<Row> test) {
        Dataset<Row> result = model.transform(test);
        Dataset<Row> predictionAndLabels = result.select("prediction", "label");
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");

        System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
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
