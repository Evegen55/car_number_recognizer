package digit.recogniser;

import digit.recogniser.ui.ProgressBar;
import digit.recogniser.ui.UI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static digit.recogniser.util.Util.setHadoopHomeEnvironmentVariable;

public class MainApp {

    private final static Logger LOGGER = LoggerFactory.getLogger(MainApp.class);

    public static final ExecutorService EXECUTOR_SERVICE = Executors.newCachedThreadPool();

    private static final JFrame MAIN_FRAME = new JFrame();

    public static void main(String... args) throws Exception {

        LOGGER.info("Setting up the HADOOP_HOME environment variable");
        setHadoopHomeEnvironmentVariable();
        LOGGER.info("The HADOOP_HOME environment variable is done");

        final ProgressBar progressBar = new ProgressBar(MAIN_FRAME, true);
        progressBar.showProgressBar("Collecting data this make take several seconds!");

        final UI ui = new UI();

        new Thread(() -> {
            try {
                ui.initUI();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                MAIN_FRAME.dispose();
            }
        }).start();

    }

}
