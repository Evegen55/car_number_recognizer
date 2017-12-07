package digit.recogniser.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class IdxReader {

    private final static Logger LOGGER = LoggerFactory.getLogger(IdxReader.class);

    private static final String INPUT_IMAGE_PATH = "resources_for_train/train-images.idx3-ubyte";
    private static final String INPUT_LABEL_PATH = "resources_for_train/train-labels.idx1-ubyte";

    private static final String INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA = "resources_for_train/t10k-images.idx3-ubyte";
    private static final String RESOURCES_FOR_TRAIN_T10K_LABELS_IDX1_UBYTE = "resources_for_train/t10k-labels.idx1-ubyte";

    public static List<LabeledImage> loadData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH, INPUT_LABEL_PATH, size);
    }

    public static List<LabeledImage> loadTestData(final int size) {
        return getLabeledImages(INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA, RESOURCES_FOR_TRAIN_T10K_LABELS_IDX1_UBYTE, size);
    }

    private static List<LabeledImage> getLabeledImages(final String inputImagePath, final String inputLabelPath, final int amountOfDataSet) {

        final List<LabeledImage> labeledImageArrayList = new ArrayList<>(amountOfDataSet);

        try (FileInputStream inImage = new FileInputStream(inputImagePath);
             FileInputStream inLabel = new FileInputStream(inputLabelPath)) {

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int numberOfRows = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            int numberOfPixels = numberOfRows * numberOfColumns;

            double[] imgPixels = new double[numberOfPixels];

            LOGGER.info("Creating ADT filed with Labeled Images ...");
            long start = System.currentTimeMillis();
            for (int i = 0; i < amountOfDataSet; i++) {

                if (i % 1000 == 0) {
                    LOGGER.info("Number of images extracted: " + i);
                }

                for (int p = 0; p < numberOfPixels; p++) {
                    imgPixels[p] = inImage.read();
                }

                int label = inLabel.read();
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