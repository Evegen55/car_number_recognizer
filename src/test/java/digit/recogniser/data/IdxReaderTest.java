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

    List<LabeledImage> labeledImages;

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void loadDataOne_Thousand() throws Exception {
        labeledImages = IdxReader.loadData(1000);
    }

    @Test
    public void loadData10_Thousand() throws Exception {
        labeledImages = IdxReader.loadData(10000);
    }

    @Test
    public void loadData60_Thousand() throws Exception {
        labeledImages = IdxReader.loadData(60000);
    }

    @Test
    public void loadTestData() throws Exception {
    }

}