/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.io.Serializable;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 *
 * @author Fanda
 */
public class FFNNTubes extends AbstractClassifier implements Serializable{
    private int nHidden; //jumlah hidden layer
    private int nNeuron; //jumlah neuron dalam hidden layer
    private int nEpoch; //jumlah iterasi
    private int nAttribute; //jumlah attribute masukan
    private int nOutput; //jumlah output masukan (= jumlah kelas)
    private int nData; //jumlah data train yang menjadi masukan
    private double learningRate; //nilai learning rate
    private double accuracy;
    
    private double[] input;
    private double[] hidden;
    private double[] output;
    
    private double[][] target;
    
    public FFNNTubes(double learn, int hid, int neuron,int iter) {
        learningRate = learn;
        nHidden = hid;
        nNeuron = neuron;
        nEpoch = iter;
    }
    
    protected static Random random = new Random();
    
    //Array of weight
    public double[][] Weight1;
    private double[][] Weight2;
    
    
    private double activationFunction (double sum) {
        double result = 1.0/(1.0+Math.exp(-sum));
        
        return result;
    }
    
    public static double randomInRange(double min, double max) {
        double range = max - min;
        double scaled = random.nextDouble() * range;
        double shifted = scaled + min;
        return shifted;
    }
    
    //setter untuk jumlah hidden layer dalam MLP
    public void setHiddenLayer(int layer) {
        this.nHidden = layer;
    }
    
    //setter untuk jumlah neuron dalam hidden layer
    public void setNeuronNum(int num) {
        this.nNeuron = num;
    }
    
    //setter untuk jumlah iterasi dalam learning
    public void setEpoch(int i) {
        this.nEpoch = i;
    }
    
    //setter untuk nilai learning rate
    public void setLearningRate(int lr) {
        this.learningRate = lr;
    }
    
    private void generateRandomWeight() {
        if (nHidden == 0) {
            Weight1 = new double[nOutput][nAttribute+1];
            for (int i = 0; i < nOutput; i++) {
                for (int j = 0; j <= nAttribute; j++) {
                    Weight1[i][j] = randomInRange(-5, 5);
                } 
            }
        } else {
            Weight1 = new double[nNeuron+1][nAttribute+1];
            Weight2 = new double[nOutput][nNeuron+1];
            for (int i = 0; i <= nNeuron; i++) {
                for (int j = 0; j <= nAttribute; j++) {
                    Weight1[i][j] = randomInRange(-5, 5);
                } 
            }
            
            for (int i = 0; i < nOutput; i++) {
                for (int j = 0; j <= nNeuron; j++) {
                    Weight2[i][j] = randomInRange(-5, 5);
                } 
            }
        }
    }
    
    private void setTarget(Instances data) {
        target = new double[nData][nOutput];
        
        for (int i = 0; i < nData; i++) {
            Instance current = data.get(i);
            for (int j = 0; j < nOutput; j++) {
                if (j == current.classValue()) {
                    target[i][j] = 1.0;
                } else {
                    target[i][j] = 0.0;
                }
            } 
        }
    }
    
    private void feedForward(Instance instance) {
        input = new double[nAttribute+1];
        input[0] = 1.0;
        for (int i = 1; i <= nAttribute; i++) {
            input[i] = (double) instance.value(i-1);
        }
        if (nHidden == 0) {
            output = new double[nOutput];
            for (int i = 0; i < nOutput; i++) {
                output[i] = 0.0;
                double sum = 0.0;
                for (int j = 0; j <= nAttribute; j++) {
                    sum = sum + (Weight1[i][j] * input[j]);
                }
                output[i] = activationFunction(sum);
            }
        } else {
            hidden = new double[nNeuron+1];
            output = new double[nOutput];
            for (int i = 0; i <= nNeuron; i++) {
                hidden[i] = 0.0;
                double sum = 0.0;
                for (int j = 0; j <= nAttribute; j++) {
                    sum = sum + (Weight1[i][j] * input[j]); 
                }
                hidden[i] = activationFunction(sum);
            }
            
            for (int i = 0; i < nOutput; i++) {
                output[i] = 0.0;
                double sum = 0.0;
                for (int j = 0; j <= nNeuron; j++) {
                    sum = sum + (Weight2[i][j] * hidden[j]); 
                }
                output[i] = activationFunction(sum);
            }
        }
    }
    
    private void updateWeight(double[] tar) {
        for (int i = 0; i < nOutput; i++) {
            for (int j = 0; j <= nAttribute; j++) {
                double tarout = (double) tar[i] - (double) output[i];
                double ti = tarout * (double) input[j];
                double temp = (double)learningRate * ti;
                Weight1[i][j] = Weight1[i][j] + (temp);
            }
        }
    }
    
    private void backPropagation(double[] tar) {
        double[] error2 = new double[nOutput]; //error untuk output
        double[] error1 = new double[nNeuron+1]; //error untuk hidden
        
        error1[0] = 0.0;
        
        for (int i = 0; i < nOutput; i++) {
            error2[i] = output[i] * (1 - output[i]) * (tar[i] - output[i]);
        }
        
        for (int i = 0; i <= nNeuron; i++) {
            double sigma = 0.0;
            for (int j = 0; j < nOutput; j++) {
                sigma = sigma + (Weight2[j][i] * error2[j]);
            }
            error1[i] = hidden[i] * (1 - hidden[i]) * sigma;
        }
        
        for (int i = 0; i < nOutput; i++) {
            for (int j = 0; j <= nNeuron; j++) {
                Weight2[i][j] = Weight2[i][j] + (learningRate * error2[i] * hidden[j]);
            }
        }
        
        for (int i = 0; i <= nNeuron; i++) {
            for (int j = 0; j <= nAttribute; j++) {
                Weight1[i][j] = Weight1[i][j] + (learningRate * error1[i] * input[j]);
            }
        }
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {        
        getCapabilities().testWithFail(data);
        
        data.deleteWithMissingClass();
        
        nAttribute = data.numAttributes()-1;
        nOutput = data.numClasses();
        nData = data.size();
        
        //set target data
        setTarget(data);
        
        //generate weight
        generateRandomWeight();
        
        //normalisasi data
        Normalize norm = new Normalize();
        Filter filter = new NominalToBinary();
        
        norm.setInputFormat(data);
        
        Instances filteredData = Filter.useFilter(data, norm);
        
        try {
            filter.setInputFormat(filteredData);
            
            for (Instance i1 : filteredData) {
                filter.input(i1);
            }
            
            filter.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        int z = 0;
        while ((z <= nEpoch) && (accuracy < 0.99)) {
            for (int j = 0; j < nData; j++) {
                feedForward(filteredData.get(j));
                
                if (nHidden == 0) {
                    updateWeight(target[j]);
                } else {
                    backPropagation(target[j]);
                }
            }
            
            countError(filteredData);
            System.out.println("ACCURACY " + z + " : " + accuracy);
            z++;
        }
    }
    
    public double[][] getWeight1 () {
        return Weight1;
    }
    
    @Override 
    public double classifyInstance(Instance instance) throws Exception {
        feedForward(instance); 
        
        double max = 0.0; 
        int maxIdx = 0; 
        int i = 0;
        while (i < output.length) {
            if (output[i] > max) {
                    max = output[i]; 
                    maxIdx = i; 
            } 
            i++; 
        }
        return (double) maxIdx; 
    }
    
    private void countError(Instances a) throws Exception {
        int error = 0;
        for (int i = 0; i < nData; i++) {
            double temp = classifyInstance(a.get(i));
            if (temp != a.get(i).classValue()) {
                error = error + 1;
            }
        }
        accuracy = ((double) nData - (double) error)/ (double)nData;
    }
}
