package digit.recogniser;

import digit.recogniser.ui.ProgressBar;
import digit.recogniser.ui.UI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainApp {

    private final static Logger LOGGER = LoggerFactory.getLogger(MainApp.class);

    public static final ExecutorService EXECUTOR_SERVICE = Executors.newCachedThreadPool();

    private static final JFrame MAIN_FRAME = new JFrame();

    public static void main(String... args) throws Exception {

        LOGGER.info("Setting up the HADOOP_HOME environment variable");
        setHadoopHomeEnvironmentVariable();
        LOGGER.info("The HADOOP_HOME environment variable is done");

        final ProgressBar progressBar = new ProgressBar(MAIN_FRAME,true);
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


    private static void setHadoopHomeEnvironmentVariable() throws Exception {
        final HashMap<String, String> hadoopEnvSetUp = new HashMap<>();
        hadoopEnvSetUp.put("HADOOP_HOME", new File("resources_for_train/winutils-master/hadoop-2.8.1").getAbsolutePath());
        final Class<?> processEnvironmentClass = Class.forName("java.lang.ProcessEnvironment");
        final Field theEnvironmentField = processEnvironmentClass.getDeclaredField("theEnvironment");
        theEnvironmentField.setAccessible(true);
        final Map<String, String> env = (Map<String, String>) theEnvironmentField.get(null);
        env.clear();
        env.putAll(hadoopEnvSetUp);
        final Field theCaseInsensitiveEnvironmentField = processEnvironmentClass.getDeclaredField("theCaseInsensitiveEnvironment");
        theCaseInsensitiveEnvironmentField.setAccessible(true);
        final Map<String, String> cienv = (Map<String, String>) theCaseInsensitiveEnvironmentField.get(null);
        cienv.clear();
        cienv.putAll(hadoopEnvSetUp);
    }
}
