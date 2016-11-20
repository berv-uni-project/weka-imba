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
    private int nAttribute; //jumlah attribute masukan
    private int nOutput; //jumlah output masukan (= jumlah kelas)
    private int nData; //jumlah data train yang menjadi masukan
    private double learningRate; //nilai learning rate
    Instances filteredData;
    
    private Double[] input;
    private Double[] hidden;
    private Double[] output;
    
    private Double[][] hiddens;
    private Double[][] outputs;
    private Double[][] target;
    
    protected static Random random = new Random();
    
    //Array of weight
    private Double[][] Weight1;
    private Double[][] Weight2;  
    
    Filter filter = new NominalToBinary();
    Normalize norm = new Normalize();
            
    public FFNN() {
        //set variable ke nilai default masing-masing
        this.nHidden = 0;
        this.nNeuron = 0;
        this.learningRate = 0.3;
        this.nEpoch = 1;
    } 
    
        @Override
    public void buildClassifier (Instances data) throws Exception {        
        nOutput = data.numClasses();
        nAttribute = data.numAttributes() - 1;
        nData = data.size();
        
        generateRandomWeight();

        /*System.out.println(nNeuron + "sesudahhhhhhhhhhhhhhhhhhhhhhhhh2222222222222222222222222222222222222");
        for (int a = 1; a <= nNeuron; a++) {
            for (int b = 1; b <= nOutput; b++) {
                System.out.println("drctasfyugxbwhcudjkwndc" + a + " " + b + " " + Weight2[a][b]);
            }
        }*/
        
        //cek kelas, bisa di-handle atau tidak
        getCapabilities().testWithFail(data);
        
        //bersihkan instances yang ada dari instances yang kelasnya miss
        data = new Instances(data);
        data.deleteWithMissingClass();
        //System.out.println(data.toString());
                
        //normalisasi
        norm.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, norm);
        
        
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
        
        setTarget(filteredData);
        /*for (int a = 1; a <= nData; a++) {
            for (int b = 1; b <= nOutput; b++) {
                System.out.println("target " + a + " " + b + " " + target[a][b]);
            }
        }*/
        
        //System.out.println("test");
        generateRandomWeight();
        hiddens = new Double[nData+1][nNeuron+1];
        outputs = new Double[nData+1][nOutput+1];
        for (int x = 1; x <= nEpoch; x++) {//iterasi utama
            //olah
            for (int i = 1; i <= nData; i++) {
                    feedForward(filteredData.instance(i-1));
                    for (int j = 1; j <= nNeuron; j++) {     
                        hiddens[i][j] = hidden[j];

                    }
                    for (int y = 1; y <= nOutput; y++) {     
                        outputs[i][y] = output[y];
                       //System.out.println("output " + output[y]);
                    }
                    if (nHidden == 0) {
                        updateWeight(filteredData.instance(i-1), target[i], outputs[i]);
                    } else if (nHidden == 1) {
                        backpropagation(filteredData.instance(i-1), target[i], outputs[i], hiddens[i]);
                    }
                
            }

            //System.out.println("outputttttttttttttttttttttttttttttttttttttt");
            /*for (int a = 1; a <= nData; a++) {
                for (int b = 1; b <= nOutput; b++) {
                    //System.out.println(a + " " + b + " " + outputs[a][b]);
                }
            }

            //System.out.println("sebelummmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm");
            for (int a = 1; a <= nAttribute; a++) {
                for (int b = 1; b <= nOutput; b++) {
                    //System.out.println(a + " " + b + " " + Weight1[a][b]);
                }
            }*/
        }
        
        /*if ( nHidden == 0) {
            System.out.println("sesudahhhhhhhhhhhhhhhhhhhhhhhhh1111111111111111111111111111111111");
            for (int a = 1; a <= nAttribute; a++) {
                for (int b = 1; b <= nOutput; b++) {
                    System.out.println(a + " " + b + " " + Weight1[a][b]);
                }
            }
        } else if (nHidden == 1) {
            System.out.println("sesudahhhhhhhhhhhhhhhhhhhhhhhhh1111111111111111111111111111111111");
            for (int a = 1; a <= nAttribute; a++) {
                for (int b = 1; b <= nNeuron; b++) {
                    System.out.println(a + " " + b + " " + Weight1[a][b]);
                }
            }

            System.out.println("sesudahhhhhhhhhhhhhhhhhhhhhhhhh2222222222222222222222222222222222222");
            for (int a = 1; a <= nNeuron; a++) {
                for (int b = 1; b <= nOutput; b++) {
                    System.out.println("uvufvhbsiubvuihbg" + a + " " + b + " " + Weight2[a][b]);
                }
            }
        }*/
        
        //hiddens = new Double[nData+1][nNeuron+1];
        //outputs = new Double[nData+1][nOutput+1];
        /*for (int i = 1; i <= nData; i++) {
            feedForward(filteredData.instance(i-1));
            for (int j = 1; j <= nNeuron; j++) {     
                hiddens[i][j] = hidden[j];

            }
            for (int j = 1; j <= nOutput; j++) {     
                outputs[i][j] = output[j];
                System.out.println("i " + i + " j " + j + " " + outputs[i][j]);
            }
            
            
        }*/
        /*for (int i = 1; i <= nData; i++) {
            System.out.println("HASIL YANG "+ (i)+" : "+classifyInstance(filteredData.instance(i-1)));
        }*/
        /*System.out.println("WEIGHT 2");
        for (int a = 1; a <= nNeuron; a++) {
            for (int b = 1; b <= nOutput; b++) {
                System.out.println(a + " " + b + " " + Weight2[a][b]);
            }
        }*/
    }
    
    private void feedForward(Instance ins) {
        input = new Double[nAttribute+1];
        for (int i = 1; i <= nAttribute; i++) {
            input[i] = ins.value(i-1);
            //System.out.println("input= " + input[i]);
        }
        
        if (nHidden == 1) {
            //olah nilai output hidden
            hidden = new Double[nNeuron+1];
            for (int a = 1; a <= nNeuron; a++) {
                Double result = 0.0;
                for (int b = 1; b <= nAttribute; b++) {
                    result = result + (input[b] * Weight1[b][a]); 
                }
                hidden[a] = activationFunction(result); 
            }
            
            //olah nilai output
            output = new Double[nOutput+1];
            for (int k = 1; k <= nOutput; k++) {
                Double result = 0.0;
                for (int l = 1; l <= nNeuron; l++) {
                    result = result + (hidden[l] * Weight2[l][k]);
                }
                output[k] = activationFunction(result);
               // System.out.println("output forward " + output[k]);
            }
        } else if (nHidden == 0) {
            //olah output
            output = new Double[nOutput+1];
            for (int k = 1; k <= nOutput; k++) {
                Double result = 0.0;
                for (int l = 1; l <= nAttribute; l++) {
                    result = result + (input[l] * Weight1[l][k]);
                }
                output[k] = activationFunction(result);
            }
        }
    }
    
    //fungsi aktivasi untuk setiap node
    private double activationFunction (Double sum) {
        double result = 1.0/(1.0+Math.exp(-sum));
        
        return result;
    }
    
    //prosedur untuk mendefinisikan bobot dari penghubung neuron
    //nilai yang di-assign antara 0.3 - 1
    private void generateRandomWeight () { 
       if (nHidden == 0) {
           Weight1 = new Double[nAttribute+1][nOutput+1];
           Weight2 = new Double[nNeuron+1][nOutput+1];
           //System.out.println("nAtt" + nAttribute);
           //System.out.println("nOut" + nOutput);
           for (int i = 1; i <= nAttribute; i++) {
                for (int j = 1; j <= nOutput; j++) {
                    Weight1[i][j] = randomInRange(0, 1);
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
                    Weight1[i][j] = randomInRange(0, 1);
                    //System.out.println(i + " " + j + " " + Weight1[i][j]);
                }
            }
            
            //System.out.println("nNeu" + nNeuron);
            //System.out.println("nOut" + nOutput);
            for (int k = 1; k <= nNeuron; k++) {
                for (int l = 1; l <= nOutput; l++) {
                    Weight2[k][l] = randomInRange(0, 1);
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
    
    public void setTarget (Instances inst) {
        int nClass = inst.numClasses();
        target = new Double[inst.size()+1][nClass+1];
        for (int i = 1; i <= inst.size(); i++) {
            System.out.println("TARGET "+ i+"HARUSNYA"+ (inst.get(i-1).classValue()));
            for (int j = 1; j <= nClass; j++) {
                if (j == (inst.get(i-1).classIndex()+1)) {
                    target[i][j] = 1.0;
                }
                else {
                    target[i][j] = 0.0;
                }
            }
        }
    }
    
    public void backpropagation (Instance inst, Double[] tar, Double[] out, Double[] hid) {
        int count = nOutput;
        Double[] errorOutput = new Double[count+1];
        Double[] errorHidden = new Double[nNeuron+1];
        
        //Error di output layer
        for (int i = 1; i <= count; i++) {
            errorOutput[i] = out[i] * (1 - out[i]) * (tar[i] - out[i]);
           // System.out.println("err, output " + errorOutput[i] + " " + out[i]);
        }
        
        //Error di hidden layer
        for (int i = 1; i <= nNeuron; i++) {
            Double sigma = 0.0;
            for (int j = 1; j <= count; j++) {
                sigma = sigma + (Weight2[i][j] * errorOutput[j]);
             //   System.out.println("hidden weight, error " + Weight2[i][j] + " " + errorOutput[j]);
            }
            //System.out.println(i);
            errorHidden[i] = hid[i] * (1 - hid[i]) * sigma;
            //System.out.println("hid, sigma " + hid[i] + " " + sigma);
        }
        
        //Update weight di hidden
        for (int i = 1; i <= nNeuron; i++) {
            for (int j = 1; j <= (nAttribute); j++) {
                Weight1[j][i] = Weight1[j][i] + (learningRate * errorHidden[i] * inst.value(j-1));
                //System.out.println("error, inst.val " + errorHidden[i] + ", " +  inst.value(j-1));
            }
        }
        
        //Update weigth di output
        for (int i = 1; i <= count; i++) {
            for (int j = 1; j <= nNeuron; j++) {
                Weight2[j][i] = Weight2[j][i] + (learningRate * errorOutput[i] * hid[j]);
            }
        }
    }
    
    public void updateWeight (Instance inst, Double[] tar, Double[] out) {
        for (int i = 1; i <= nOutput; i++) {
            for (int j = 1; j <= nAttribute; j++) {
                //System.out.println("i " + i + "j " + j + "ins value " + inst.value(j-1));
                //System.out.println("Weight AWALLLLLLLLLLLLLLLLLLL= " + Weight1[j][i]);
                //System.out.println("TARRRRRRRRRRRRRRRRRRRRR= " + tar[i]);
                //System.out.println("OUTTTTTTTTTTTTTTTTTTTTT= " + out[i]);
                Weight1[j][i] = Weight1[j][i] + (learningRate * (tar[i] - out[i]) * inst.value(j-1));
                //System.out.println("Weight AKHIRRRRRRRRRRRRRRRR= " + Weight1[j][i]);
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
               
        /*norm.input(instance);
        Instance baru = norm.output();
        
        filter.input(baru);
        Instance curr_ins = filter.output();
        
        feedForward(curr_ins);*/
        
        feedForward(instance);

        double max = 0.0;
        int maxIdx = 0;
        int i = 1;
        while (i < output.length) {
                if (output[i] > max) {
                        max = output[i];
                        maxIdx = i;
                }
                i++;
        }
        
        return maxIdx;
    }
}