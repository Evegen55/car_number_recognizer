package digit.recogniser.data;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * @author (created on 12/7/2017).
 */
public class IdxReaderTest {

    private IdxReader idxReader;

    @Before
    public void setUp() throws Exception {
        idxReader = new IdxReader();
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void loadData() throws Exception {
        List<LabeledImage> labeledImages = idxReader.loadData(10000);
    }

    @Test
    public void loadTestData() throws Exception {
    }

}