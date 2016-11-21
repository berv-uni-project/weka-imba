/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ui;

import imba.classifier.FFNNTubes;
import imba.classifier.NBTubes;
import java.io.File;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.plaf.metal.MetalLookAndFeel;
import javax.swing.plaf.metal.OceanTheme;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Bervianto Leo P
 */
public class MainWindow extends javax.swing.JFrame {

    /**
     * Creates new form MainWindow
     */
    public MainWindow() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        horizontalFillLeft = new javax.swing.Box.Filler(new java.awt.Dimension(20, 0), new java.awt.Dimension(30, 0), new java.awt.Dimension(50, 0));
        allPanel = new javax.swing.JPanel();
        mainPanel = new javax.swing.JPanel();
        openButton = new javax.swing.JButton();
        saveButton = new javax.swing.JButton();
        loadModelButton = new javax.swing.JButton();
        saveModelButton = new javax.swing.JButton();
        dataTestButton = new javax.swing.JButton();
        datasetStatusPanel = new javax.swing.JPanel();
        relationLabel = new javax.swing.JLabel();
        relationValue = new javax.swing.JLabel();
        attributesLabel = new javax.swing.JLabel();
        attributesValue = new javax.swing.JLabel();
        instancesLabel = new javax.swing.JLabel();
        instancesValue = new javax.swing.JLabel();
        sumofweightsLabel = new javax.swing.JLabel();
        sumofweightsValue = new javax.swing.JLabel();
        selectMethodePanel = new javax.swing.JPanel();
        selectClassifierBox = new javax.swing.JComboBox<>();
        selectEvaluationBox = new javax.swing.JComboBox<>();
        FFNNProperties = new javax.swing.JPanel();
        iterationLabel = new javax.swing.JLabel();
        iterationValue = new javax.swing.JTextField();
        learningRateLabel = new javax.swing.JLabel();
        learningRateValue = new javax.swing.JTextField();
        neuronLabel = new javax.swing.JLabel();
        neuronValue = new javax.swing.JTextField();
        hiddenLayerLabel = new javax.swing.JLabel();
        hiddenLayerValue = new javax.swing.JTextField();
        runningPane = new javax.swing.JPanel();
        resultLabel = new javax.swing.JLabel();
        resultPane = new javax.swing.JScrollPane();
        resultTextArea = new javax.swing.JTextArea();
        executeButton = new javax.swing.JButton();
        statusPanel = new javax.swing.JPanel();
        statusLabel = new javax.swing.JLabel();
        statusValue = new javax.swing.JLabel();
        horizontalFillRight = new javax.swing.Box.Filler(new java.awt.Dimension(20, 0), new java.awt.Dimension(30, 0), new java.awt.Dimension(50, 0));
        modelLabel = new javax.swing.JLabel();
        modelValue = new javax.swing.JLabel();
        mainMenuBar = new javax.swing.JMenuBar();
        fileMenu = new javax.swing.JMenu();
        exitFileMenuItem = new javax.swing.JMenuItem();
        aboutMenu = new javax.swing.JMenu();
        aboutHelpMenuItem = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Weka Imba");
        getContentPane().setLayout(new javax.swing.BoxLayout(getContentPane(), javax.swing.BoxLayout.X_AXIS));
        getContentPane().add(horizontalFillLeft);

        allPanel.setLayout(new javax.swing.BoxLayout(allPanel, javax.swing.BoxLayout.Y_AXIS));

        openButton.setText("Open Dataset File");
        openButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openButtonActionPerformed(evt);
            }
        });
        mainPanel.add(openButton);

        saveButton.setText("Save Dataset");
        saveButton.setEnabled(false);
        mainPanel.add(saveButton);

        loadModelButton.setText("Load Model");
        loadModelButton.setEnabled(false);
        loadModelButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadModelButtonActionPerformed(evt);
            }
        });
        mainPanel.add(loadModelButton);

        saveModelButton.setText("Save Model");
        saveModelButton.setEnabled(false);
        saveModelButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveModelButtonActionPerformed(evt);
            }
        });
        mainPanel.add(saveModelButton);

        dataTestButton.setText("Input Data Test");
        dataTestButton.setEnabled(false);
        dataTestButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                dataTestButtonActionPerformed(evt);
            }
        });
        mainPanel.add(dataTestButton);

        allPanel.add(mainPanel);

        datasetStatusPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Data Set Status"));
        datasetStatusPanel.setLayout(new java.awt.GridLayout(2, 2));

        relationLabel.setText("Relation :");
        datasetStatusPanel.add(relationLabel);
        datasetStatusPanel.add(relationValue);

        attributesLabel.setText("Attributes :");
        datasetStatusPanel.add(attributesLabel);
        datasetStatusPanel.add(attributesValue);

        instancesLabel.setText("Instances :");
        datasetStatusPanel.add(instancesLabel);
        datasetStatusPanel.add(instancesValue);

        sumofweightsLabel.setText("Sum of Weights :");
        datasetStatusPanel.add(sumofweightsLabel);
        datasetStatusPanel.add(sumofweightsValue);

        allPanel.add(datasetStatusPanel);

        selectMethodePanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Methode Properties"));

        selectClassifierLabel.setText("Select Classifier : ");
        selectMethodePanel.add(selectClassifierLabel);

        selectClassifierBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "FFNN", "Naive Bayes" }));
        selectClassifierBox.setEnabled(false);
        selectMethodePanel.add(selectClassifierBox);

        selectEvaluationLabel.setText("Select Evaluation :");
        selectMethodePanel.add(selectEvaluationLabel);

        selectEvaluationBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Cross Validation", "Split Test" }));
        selectEvaluationBox.setEnabled(false);
        selectMethodePanel.add(selectEvaluationBox);

        allPanel.add(selectMethodePanel);

        FFNNProperties.setBorder(javax.swing.BorderFactory.createTitledBorder("FFNN Properties"));
        FFNNProperties.setName("FFNN Properties"); // NOI18N
        FFNNProperties.setLayout(new java.awt.GridLayout(2, 4, 5, 5));

        iterationLabel.setText("Jumlah Iterasi :");
        FFNNProperties.add(iterationLabel);

        iterationValue.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        iterationValue.setPreferredSize(new java.awt.Dimension(60, 20));
        FFNNProperties.add(iterationValue);

        learningRateLabel.setText("Learning Rate :");
        FFNNProperties.add(learningRateLabel);

        learningRateValue.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        learningRateValue.setPreferredSize(new java.awt.Dimension(30, 20));
        FFNNProperties.add(learningRateValue);

        neuronLabel.setText("Jumlah Neuron :");
        FFNNProperties.add(neuronLabel);

        neuronValue.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        neuronValue.setPreferredSize(new java.awt.Dimension(30, 20));
        FFNNProperties.add(neuronValue);

        hiddenLayerLabel.setText("Jumlah Hidden Layer :");
        FFNNProperties.add(hiddenLayerLabel);

        hiddenLayerValue.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        hiddenLayerValue.setPreferredSize(new java.awt.Dimension(30, 20));
        FFNNProperties.add(hiddenLayerValue);

        allPanel.add(FFNNProperties);

        runningPane.setBorder(javax.swing.BorderFactory.createTitledBorder("Result Panel"));
        runningPane.setLayout(new javax.swing.BoxLayout(runningPane, javax.swing.BoxLayout.LINE_AXIS));

        resultLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        resultLabel.setText("Result :");
        runningPane.add(resultLabel);

        resultTextArea.setEditable(false);
        resultTextArea.setColumns(20);
        resultTextArea.setRows(5);
        resultPane.setViewportView(resultTextArea);

        runningPane.add(resultPane);

        executeButton.setText("Execute");
        executeButton.setEnabled(false);
        executeButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                executeButtonActionPerformed(evt);
            }
        });
        runningPane.add(executeButton);

        allPanel.add(runningPane);

        statusPanel.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.LOWERED, java.awt.Color.black, java.awt.Color.white, java.awt.Color.black, java.awt.Color.white));

        statusLabel.setText("Working Status :");
        statusPanel.add(statusLabel);

        statusValue.setText("Idle");
        statusPanel.add(statusValue);
        statusPanel.add(horizontalFillRight);

        modelLabel.setText("Model Status :");
        statusPanel.add(modelLabel);

        modelValue.setText("Not Available");
        statusPanel.add(modelValue);

        allPanel.add(statusPanel);

        getContentPane().add(allPanel);

        fileMenu.setText("File");

        exitFileMenuItem.setText("Exit");
        exitFileMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                exitFileMenuItemActionPerformed(evt);
            }
        });
        fileMenu.add(exitFileMenuItem);

        mainMenuBar.add(fileMenu);

        aboutMenu.setText("Help");

        aboutHelpMenuItem.setText("About");
        aboutMenu.add(aboutHelpMenuItem);

        mainMenuBar.add(aboutMenu);

        setJMenuBar(mainMenuBar);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void exitFileMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_exitFileMenuItemActionPerformed
        System.exit(0);
    }//GEN-LAST:event_exitFileMenuItemActionPerformed

    private void openButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_openButtonActionPerformed
        if(evt.getSource() == this.openButton) {
            this.filechooser.setAcceptAllFileFilterUsed(false);
            this.filechooser.removeChoosableFileFilter(modelformat);
            this.filechooser.setFileFilter(arffformat);
            int returnVal = this.filechooser.showOpenDialog(MainWindow.this);
            
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = this.filechooser.getSelectedFile();
                    this.statusValue.setText("Membuka: " + file.getName() + ".\n");
                    this.data = ConverterUtils.DataSource.read(file.getAbsolutePath());
                    // Minta input nama kelas
                    JFrame frame = new JFrame("Class Name");
                    String className = JOptionPane.showInputDialog(frame, "Class Name");
                    if (className == null) {
                        this.data.setClassIndex(this.data.attribute("class").index());
                    } else {
                        this.data.setClassIndex(this.data.attribute(className).index());
                    }
                    this.instancesValue.setText(String.valueOf(this.data.numInstances()));
                    this.attributesValue.setText(String.valueOf(this.data.numAttributes()));
                    this.relationValue.setText(String.valueOf(this.data.relationName()));
                    this.sumofweightsValue.setText(String.valueOf(this.data.sumOfWeights()));
                    this.saveButton.setEnabled(true);
                    this.executeButton.setEnabled(true);
                    this.selectEvaluationBox.setEnabled(true);
                    this.selectClassifierBox.setEnabled(true);
                    this.loadModelButton.setEnabled(true);
                    this.statusValue.setText("Membuka berkas "+file.getName()+" berhasil!");
                } catch (Exception ex) {
                    Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                this.statusValue.setText("Open command cancelled by user.\n");
            }
        }
    }//GEN-LAST:event_openButtonActionPerformed

    private void executeButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_executeButtonActionPerformed
        if(evt.getSource()==this.executeButton) {
            switch (this.selectClassifierBox.getSelectedIndex()) {
                case 0:
                    try {
                        // FFNN Section
                        double learningRate = Double.valueOf(this.learningRateValue.getText());
                        int hiddenLayer = Integer.valueOf(this.hiddenLayerValue.getText());
                        int jumlahNeuron = Integer.valueOf(this.neuronValue.getText());
                        int jumlahIterasi = Integer.valueOf(this.iterationValue.getText());
                        Classifier fn;
                        if ( (learningRate > 0.0 && learningRate < 1.0) && (hiddenLayer >= 0 && hiddenLayer <= 1) && (jumlahNeuron >= 0) && (jumlahIterasi > 0)) {
                            // Valid Input
                            fn = new FFNNTubes(learningRate,hiddenLayer,jumlahNeuron,jumlahIterasi);
                        } else {
                            // Default Running
                            fn = new FFNNTubes(0.9,1,20,100000);
                        }
                        fn.buildClassifier(this.data);
                        this.loadedModel = fn;
                        Instances ueval1 = this.data;
                        Normalize norm = new Normalize();
                        Filter filter = new NominalToBinary();
                        norm.setInputFormat(ueval1);
                        
                        Instances ueval = Filter.useFilter(ueval1, norm);
                        filter.setInputFormat(ueval);
                        
                        for (Instance i1 : ueval) {
                            filter.input(i1);
                        }
                        filter.batchFinished();
                        
                        // Evaluasi
                        switch (this.selectEvaluationBox.getSelectedIndex()) {
                            case 0:
                                Evaluation eval = new Evaluation(ueval);
                                eval.evaluateModel(fn, ueval);
                                this.resultTextArea.setText(eval.toSummaryString("\n== Summary ==\n",false));
                                this.resultTextArea.append(eval.toClassDetailsString("\n== Detailed Accuracy By Class ==\n"));
                                this.resultTextArea.append(eval.toMatrixString("\n== Confusion Matrix ==\n"));
                                this.statusValue.setText("Running Cross Validation with FFNN Model Completed");
                                break;
                            case 1:
                                break;
                            default:
                                this.statusValue.setText("Do Nothing!");
                                break;
                        }
                        this.modelValue.setText("FFNN Model");
                        this.saveModelButton.setEnabled(true);
                        this.dataTestButton.setEnabled(true);
                    } catch (Exception ex) {
                        Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                    }   break;
                case 1:
                    try {
                        // Naive Bayes Section
                        Classifier nb = new NBTubes();
                        nb.buildClassifier(this.data);
                        this.loadedModel = nb;
                        
                        // Evaluasi
                        switch (this.selectEvaluationBox.getSelectedIndex()) {
                            case 0:
                                Evaluation eval = new Evaluation(this.data);
                                eval.evaluateModel(nb, this.data);
                                this.resultTextArea.setText(eval.toSummaryString("\n== Summary ==\n",false));
                                this.resultTextArea.append(eval.toClassDetailsString("\n== Detailed Accuracy By Class ==\n"));
                                this.resultTextArea.append(eval.toMatrixString("\n== Confusion Matrix ==\n"));
                                this.statusValue.setText("Running Cross Validation with NB Completed");
                                break;
                            case 1:
                                break;
                            default:
                                this.statusValue.setText("Do Nothing!");
                                break;
                        }
                        this.modelValue.setText("Bayes Model");
                        this.saveModelButton.setEnabled(true);
                        this.dataTestButton.setEnabled(true);
                    } catch (Exception ex) {
                        Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                    }   break;
                default:
                    this.statusValue.setText("Do Nothing!");
                    break;
            }
        }
    }//GEN-LAST:event_executeButtonActionPerformed

    private void loadModelButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadModelButtonActionPerformed
        if (evt.getSource() == this.loadModelButton) {
            this.filechooser.setAcceptAllFileFilterUsed(false);
            this.filechooser.removeChoosableFileFilter(arffformat);
            this.filechooser.setFileFilter(modelformat);
            int returnVal = this.filechooser.showOpenDialog(MainWindow.this);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = this.filechooser.getSelectedFile();
                    this.statusValue.setText("Load model: " + file.getName() + ".\n");
                    loadedModel = (Classifier) SerializationHelper.read(file.getAbsolutePath());
                    this.modelValue.setText("Model "+file.getName());
                    this.dataTestButton.setEnabled(true);
                    this.saveModelButton.setEnabled(true);
                } catch (Exception ex) {
                    Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                this.statusValue.setText("Load model canceled by user.\n");
            }
        }
    }//GEN-LAST:event_loadModelButtonActionPerformed

    private void saveModelButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveModelButtonActionPerformed
        if (evt.getSource() == this.saveModelButton) {
            this.filechooser.setAcceptAllFileFilterUsed(false);
            this.filechooser.removeChoosableFileFilter(arffformat);
            this.filechooser.setFileFilter(modelformat);
            int returnVal = this.filechooser.showSaveDialog(MainWindow.this);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = this.filechooser.getSelectedFile();
                    SerializationHelper.write(file.getAbsolutePath(), this.loadedModel);
                    this.statusValue.setText("Save model completed.");
                } catch (Exception ex) {
                    Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                this.statusValue.setText("Save model canceled by user.\n");
            }
        }
    }//GEN-LAST:event_saveModelButtonActionPerformed

    private void dataTestButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_dataTestButtonActionPerformed
        if(evt.getSource() == this.dataTestButton) {
            this.filechooser.setAcceptAllFileFilterUsed(false);
            this.filechooser.removeChoosableFileFilter(modelformat);
            this.filechooser.setFileFilter(arffformat);
            int returnVal = this.filechooser.showOpenDialog(MainWindow.this);
            
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = this.filechooser.getSelectedFile();
                    this.statusValue.setText("Membuka: " + file.getName() + ".\n");
                    Instances testData = ConverterUtils.DataSource.read(file.getAbsolutePath());
                    int i = 1;
                    this.resultTextArea.setText("");
                    for(Instance test:testData) {
                        double result = loadedModel.classifyInstance(test);
                        this.resultTextArea.append("Data-"+i+" Result : "+this.data.classAttribute().value((int)result)+"\n");
                        i++;
                    }

                    this.statusValue.setText("Test berkas "+file.getName()+" berhasil!");
                } catch (Exception ex) {
                    Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                this.statusValue.setText("Open command cancelled by user.\n");
            }
        }
    }//GEN-LAST:event_dataTestButtonActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Metal look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Metal".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    MetalLookAndFeel.setCurrentTheme(new OceanTheme());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new MainWindow().setVisible(true);
            }
        });
    }
    // Variables Data
    private Instances data;
    private Classifier loadedModel;
    private final JFileChooser filechooser = new JFileChooser();
    private final ArffFile arffformat = new ArffFile();
    private final ModelFile modelformat = new ModelFile();
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel FFNNProperties;
    private javax.swing.JMenuItem aboutHelpMenuItem;
    private javax.swing.JMenu aboutMenu;
    private javax.swing.JPanel allPanel;
    private javax.swing.JLabel attributesLabel;
    private javax.swing.JLabel attributesValue;
    private javax.swing.JButton dataTestButton;
    private javax.swing.JPanel datasetStatusPanel;
    private javax.swing.JButton executeButton;
    private javax.swing.JMenuItem exitFileMenuItem;
    private javax.swing.JMenu fileMenu;
    private javax.swing.JLabel hiddenLayerLabel;
    private javax.swing.JTextField hiddenLayerValue;
    private javax.swing.Box.Filler horizontalFillLeft;
    private javax.swing.Box.Filler horizontalFillRight;
    private javax.swing.JLabel instancesLabel;
    private javax.swing.JLabel instancesValue;
    private javax.swing.JLabel iterationLabel;
    private javax.swing.JTextField iterationValue;
    private javax.swing.JLabel learningRateLabel;
    private javax.swing.JTextField learningRateValue;
    private javax.swing.JButton loadModelButton;
    private javax.swing.JMenuBar mainMenuBar;
    private javax.swing.JPanel mainPanel;
    private javax.swing.JLabel modelLabel;
    private javax.swing.JLabel modelValue;
    private javax.swing.JLabel neuronLabel;
    private javax.swing.JTextField neuronValue;
    private javax.swing.JButton openButton;
    private javax.swing.JLabel relationLabel;
    private javax.swing.JLabel relationValue;
    private javax.swing.JLabel resultLabel;
    private javax.swing.JScrollPane resultPane;
    private javax.swing.JTextArea resultTextArea;
    private javax.swing.JPanel runningPane;
    private javax.swing.JButton saveButton;
    private javax.swing.JButton saveModelButton;
    private javax.swing.JComboBox<String> selectClassifierBox;
    private final javax.swing.JLabel selectClassifierLabel = new javax.swing.JLabel();
    private javax.swing.JComboBox<String> selectEvaluationBox;
    private final javax.swing.JLabel selectEvaluationLabel = new javax.swing.JLabel();
    private javax.swing.JPanel selectMethodePanel;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JPanel statusPanel;
    private javax.swing.JLabel statusValue;
    private javax.swing.JLabel sumofweightsLabel;
    private javax.swing.JLabel sumofweightsValue;
    // End of variables declaration//GEN-END:variables
}
