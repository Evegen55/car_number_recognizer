package digit.recogniser.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class IdxReader {

    private final static Logger LOGGER = LoggerFactory.getLogger(IdxReader.class);

    public static final String INPUT_IMAGE_PATH = "resources_for_train/train-images.idx3-ubyte";
    public static final String INPUT_LABEL_PATH = "resources_for_train/train-labels.idx1-ubyte";

    public static final String INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA = "resources_for_train/t10k-images.idx3-ubyte";
    public static final String RESOURCES_FOR_TRAIN_T10K_LABELS_IDX1_UBYTE = "resources_for_train/t10k-labels.idx1-ubyte";

    public static final int VECTOR_DIMENSION = 784; //square 28*28 as from data set -> array 784 items

    /**
     * @param size
     * @return
     */
    public static List<LabeledImage> loadData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH, INPUT_LABEL_PATH, size);
    }

    /**
     * @param size
     * @return
     */
    public static List<LabeledImage> loadTestData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA, RESOURCES_FOR_TRAIN_T10K_LABELS_IDX1_UBYTE, size);
    }

    private static List<LabeledImage> getLabeledImages(final String inputImagePath,
                                                       final String inputLabelPath,
                                                       final int amountOfDataSet) {

        final List<LabeledImage> labeledImageArrayList = new ArrayList<>(amountOfDataSet);

        try (FileInputStream inImage = new FileInputStream(inputImagePath);
             FileInputStream inLabel = new FileInputStream(inputLabelPath)) {

            // just skip the amount of a data
            // see the test and description for dataset
            inImage.skip(16);
            inLabel.skip(8);
            LOGGER.debug("Available bytes in inputImage stream after read: " + inImage.available());
            LOGGER.debug("Available bytes in inputLabel stream after read: " + inLabel.available());

            //empty array for 784 pixels - the image from 28x28 pixels in a single row
            double[] imgPixels = new double[VECTOR_DIMENSION];

            LOGGER.info("Creating ADT filed with Labeled Images ...");
            long start = System.currentTimeMillis();
            for (int i = 0; i < amountOfDataSet; i++) {

                if (i % 1000 == 0) {
                    LOGGER.info("Number of images extracted: " + i);
                }
                //it fills the array of pixels
                for (int index = 0; index < VECTOR_DIMENSION; index++) {
                    imgPixels[index] = inImage.read();
                }
                //it creates a label for that
                int label = inLabel.read();
                //it creates a compound object and adds them to a list
                labeledImageArrayList.add(new LabeledImage(label, imgPixels));
            }
            LOGGER.info("Time to load LabeledImages in seconds: " + ((System.currentTimeMillis() - start) / 1000d));
        } catch (Exception e) {
            LOGGER.error("Smth went wrong: \n");
            e.printStackTrace();
        }

        return labeledImageArrayList;
    }

}