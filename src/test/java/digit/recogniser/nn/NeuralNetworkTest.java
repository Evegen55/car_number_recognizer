package digit.recogniser.nn;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static digit.recogniser.util.Util.setHadoopHomeEnvironmentVariable;

/**
 * @author (created on 12/7/2017).
 */
public class NeuralNetworkTest {

    private NeuralNetwork NEURAL_NETWORK;

    private static final int[] LAYERS_BIGGER = new int[]{784, 2500, 2000, 1500, 1000, 500, 128, 64, 10};

    @Before
    public void setUp() throws Exception {
        setHadoopHomeEnvironmentVariable();
        NEURAL_NETWORK = new NeuralNetwork();
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void init() throws Exception {
    }

    @Test
    public void train_30000_10000() throws Exception {
        NEURAL_NETWORK.train(30000, 10000, true);
    }

    @Test
    public void train_60000_10000() throws Exception {
        NEURAL_NETWORK.train(60000, 10000, true);
    }

    @Test
    public void predict() throws Exception {
    }

}