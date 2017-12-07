package digit.recogniser.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class IdxReader {

    private final static Logger LOGGER = LoggerFactory.getLogger(IdxReader.class);

    private static final String INPUT_IMAGE_PATH = "resources_for_train/train-images.idx3-ubyte";
    private static final String INPUT_LABEL_PATH = "resources_for_train/train-labels.idx1-ubyte";

    private static final String INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA = "resources_for_train/t10k-images.idx3-ubyte";
    private static final String INPUT_LABEL_PATH_FOR_LOADING_TEST_DATA = "resources_for_train/t10k-labels.idx1-ubyte";

    public java.util.List<LabeledImage> loadData(int size) {
        return getLabeledImages(INPUT_IMAGE_PATH, INPUT_LABEL_PATH, size);
    }

    public java.util.List<LabeledImage> loadTestData(int size) {
        return getLabeledImages(INPUT_IMAGE_PATH_FOR_LOADING_TEST_DATA, INPUT_LABEL_PATH_FOR_LOADING_TEST_DATA, size);
    }

    private List<LabeledImage> getLabeledImages(final String inputImagePath, final String inputLabelPath, final int number) {

        //create empty ADT for given amount of data
        final List<LabeledImage> labeledImages = new ArrayList<>(number);

        try (FileInputStream inLabel = new FileInputStream(inputImagePath);
             FileInputStream inImage = new FileInputStream(inputLabelPath)) {

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int numberOfRows = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            int numberOfPixels = numberOfRows * numberOfColumns;

            LOGGER.info("Number of numberOfRows: " + numberOfRows + "\n");
            LOGGER.info("Number of numberOfColumns: " + numberOfColumns + "\n");
            LOGGER.info("Number of pixels: " + numberOfPixels + "\n");

            double[] imgPixels = new double[numberOfPixels];


            LOGGER.info("Work is started");
            long start = System.currentTimeMillis();
            for (int i = 0; i < number; i++) {

                // TODO: 12/6/2017 remove this code
                if (i % 1000 == 0) {
                    LOGGER.info("Number of images extracted: " + i);
                }

                for (int p = 0; p < numberOfPixels; p++) {
                    imgPixels[p] = inImage.read();
                }

                int label = inLabel.read();
                labeledImages.add(new LabeledImage(label, imgPixels));
            }
            LOGGER.info("Time to load LabeledImages in seconds: " + ((System.currentTimeMillis() - start) / 1000d));


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labeledImages;
    }

}