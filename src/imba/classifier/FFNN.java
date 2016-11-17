/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.io.Serializable;
import java.util.ArrayList;
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
    private int nHidden;
    private int nNeuron;
    //private double momentum;
    private int nEpoch;
    private double learningRate;
    
    //learning rate sementara definisi sendiri
        
            
    public FFNN() {
        this.nHidden = 0;
        this.nNeuron = 0;
        //this.momentum = 0.2;
        this.learningRate = 0.3;
        this.nEpoch = 500;
    } 
    
        @Override
    public void buildClassifier (Instances data) throws Exception {
        //cek kelas, bisa di-handle atau tidak
        getCapabilities().testWithFail(data);
        
        //bersihkan instances yang ada dari instances yang kelasnya miss
        data = new Instances(data);
        data.deleteWithMissingClass();
        System.out.println(data.toString());
                
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
        
        //System.out.println(filteredData.toString());
        
        //jumlah atribut masukan
        int numAttr = data.get(0).numAttributes();
    }  
}