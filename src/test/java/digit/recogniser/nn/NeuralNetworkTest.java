package digit.recogniser.nn;

import org.junit.Before;
import org.junit.Test;

import static digit.recogniser.data.IdxReader.VECTOR_DIMENSION;
import static digit.recogniser.nn.NeuralNetwork.NEURAL_OUTPUT_CLASSES;

/**
 * @author (created on 12 / 7 / 2017).
 */
public class NeuralNetworkTest {

    private NeuralNetwork NEURAL_NETWORK;

    private static final int[] LAYERS_BIGGER = new int[]{VECTOR_DIMENSION, 1024, 512, 128, 64, NEURAL_OUTPUT_CLASSES};

    @Before
    public void setUp() {
//        setHadoopHomeEnvironmentVariable(); //For Windows only
        NEURAL_NETWORK = new NeuralNetwork();
    }

    @Test
    public void train_1000d_1000t_withcustomlayers() {
        NEURAL_NETWORK.train(1000, 1000, true, LAYERS_BIGGER);
    }

    @Test
    public void train_30000_10000() {
        NEURAL_NETWORK.train(30000, 10000, true, null);
    }

    @Test
    public void train_60000_10000() {
        NEURAL_NETWORK.train(60000, 10000, true, null);
    }

    @Test
    public void train_60000_10000_withcustomlayers() {
        NEURAL_NETWORK.train(60000, 10000, true, LAYERS_BIGGER);
    }

}