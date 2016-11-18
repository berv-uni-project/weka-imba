/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Bervianto Leo P
 */
public class FFNN extends AbstractClassifier implements Serializable {
    //input pengguna untuk pengaturan topologi
    private int nHidden; //jumlah hidden layer
    private int nNeuron; //jumlah neuron dalam hidden layer
    private int nEpoch; //jumlah iterasi
    private int nAttribute; //ganti dengan nilai sebenarnya dari instances
    private int nOutput; //ganti dengan nilai sebenarnya dari instances
    private double learningRate; //nilai learning rate
    
    protected static Random random = new Random();
    
    //Array of weight
    private Double[][] Weight1;
    private Double[][] Weight2;                
            
    public FFNN() {
        //set variable ke nilai default masing-masing
        this.nHidden = 0;
        this.nNeuron = 0;
        this.learningRate = 0.3;
        this.nEpoch = 500;
        //this.nAttribute = 3;
    } 
    
        @Override
    public void buildClassifier (Instances data) throws Exception {        
        nOutput = data.numClasses();
        nAttribute = data.numAttributes() - 1;
        
        generateRandomWeight();
        
        //cek kelas, bisa di-handle atau tidak
        getCapabilities().testWithFail(data);
        
        //bersihkan instances yang ada dari instances yang kelasnya miss
        data = new Instances(data);
        data.deleteWithMissingClass();
        //System.out.println(data.toString());
                
        //normalisasi
        Normalize norm = new Normalize();
        norm.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, norm);
        
        Filter filter = new NominalToBinary();
        //filter.setInputFormat(filteredData);
        try {
            filter.setInputFormat(filteredData);
            
            for (Instance i1 : filteredData) {
                filter.input(i1);
            }
            
            filter.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }           
    }
    
    //prosedur untuk mendefinisikan bobot dari penghubung neuron
    //nilai yang di-assign antara 0.3 - 1
    private void generateRandomWeight () { 
       if (nHidden == 0) {
           Weight1 = new Double[nAttribute+1][nOutput+1];
           //System.out.println("nAtt" + nAttribute);
           //System.out.println("nOut" + nOutput);
           for (int i = 1; i <= nAttribute; i++) {
                for (int j = 1; j <= nOutput; j++) {
                    Weight1[i][j] = randomInRange(-1, 1);
                    //System.out.println(i + " " + j + " " + Weight1[i][j]);
                }
           }
           
        } else if (nHidden == 1) {
            Weight1 = new Double[nAttribute+1][nNeuron+1];
            Weight2 = new Double[nNeuron+1][nOutput+1];
            //System.out.println("nAtt" + nAttribute);
            //System.out.println("nNeu" + nNeuron);
            for (int i = 1; i <= nAttribute; i++) {
                for (int j = 1; j <= nNeuron; j++) {
                    Weight1[i][j] = randomInRange(-1, 1);
                    //System.out.println(i + " " + j + " " + Weight1[i][j]);
                }
            }
            
            //System.out.println("nNeu" + nNeuron);
            //System.out.println("nOut" + nOutput);
            for (int k = 1; k <= nNeuron; k++) {
                for (int l = 1; l <= nOutput; l++) {
                    Weight2[k][l] = randomInRange(-1, 1);
                    //System.out.println(k + " " + l + " " + Weight2[k][l]);
                }
            }
        }
    }
    
    public static double randomInRange(double min, double max) {
        double range = max - min;
        double scaled = random.nextDouble() * range;
        double shifted = scaled + min;
        return shifted; // == (rand.nextDouble() * (max-min)) + min;
    }   
    
    //setter untuk jumlah hidden layer dalam MLP
    //nilai default = 0
    public void setHiddenLayer(int layer) {
        this.nHidden = layer;
    }
    
    //setter untuk jumlah neuron dalam hidden layer
    //nilai default = 0
    public void setNeuronNum(int num) {
        this.nNeuron = num;
    }
    
    //setter untuk jumlah iterasi dalam learning
    //nilai default = 500
    public void setEpoch(int i) {
        this.nEpoch = i;
    }
    
    //setter untuk nilai learning rate
    //nilai default = 0.3
    public void setLearningRate(int lr) {
        this.learningRate = lr;
    }
}