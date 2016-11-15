/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifierTubes;

import java.io.File;
import java.util.ArrayList;
import javafx.stage.FileChooser;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;

/**
 *
 * @author absol
 */
public class NBTubes implements Classifier {
    
    public ArrayList<ArrayList<ArrayList<Integer>>> dataClassifier;
    //Urutan: 1. Atribut, 2. Domain
    //Kelas dan domain beserta jumlah instance nya dijumlah dari setiap data
    //domain dari sebuah atribut
    
    public NBTubes() {
        dataClassifier = null;
    }
    
    @Override
    public void buildClassifier(Instances data) {
        int i, j, k, l;
        
        int numAttr = data.get(0).numAttributes();
        //data.get(0).attribute(i).numValues() ini jumlah tipe nilai tiap atribut ke i
        int numClasses = data.get(0).attribute(numAttr-1).numValues();
        
        //building data structure
        dataClassifier = new ArrayList<>();
        
        i = 0;
        while (i < (numAttr-1)){
            //add new attribute, blm ada domainnya
            dataClassifier.add(new ArrayList<>());
            
            j = 0;
            while (j < data.get(0).attribute(i).numValues()) {
                //add new domain pada suatu atribut, blm ada perbedaan kelas
                dataClassifier.get(i).add(new ArrayList<>());
                
                k = 0;
                while (k < data.get(0).attribute(numAttr-1).numValues()) {
                    //add new kelas di dalam atribut, domain spesifik                    
                    dataClassifier.get(i).get(j).add(0);
                    k++;
                }
                
                j++;
            }
            
            i++;
        }
        
        //reading from instances
        i = 0;
        while (i < data.size()) {
            
            
            j = 0;
            while (j < numAttr-1) {
                
                dataClassifier.get(j).get((int) data.get(i).value(j)).
                        set((int) data.get(i).value(numAttr-1), dataClassifier.get(j).get((int) data.get(i).value(j)).get((int) data.get(i).value(numAttr-1))+1);
                
                /*
                k = 0;
                while (k < data.get(0).attribute(j).numValues()) {
                    
                    l = 0;                    
                    while (l < numClasses) {   
                        System.out.println("Sum pada atribut " + j + ", nilai " + k + ", kelas " + l + ": " + dataClassifier.get(j).get(k).get(l));
                        
                        if (data.get(i).value(numAttr-1) == l) {
                            dataClassifier.get(j).get(k).set(l, (dataClassifier.get(j).get(k).get(l) + 1));
                        }
                        
                        l++;
                    }
                    
                    k++;
                }
                */
                
                j++;
            }
            i++;
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //Fungsi ini menentukan probabilitas setiap kelas instance untuk instance 
        //yang ada di parameter fungsi
        
        double[] a = new double[dataClassifier.size()];
        
        return a;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //Fungsi ini mengembalikan indeks kelas dengan probabilitas tertinggi
        
        //panggil distributionForInstance
        
        //cari max value, return indeks kelas paling tinggi wqwqw
        
        return 0;
    }
    
    @Override
    public Capabilities getCapabilities() {
        //Fungsi ini mengembalikan "Capabilities", yaitu handler kasus2 aneh pada
        //instances training (?)
        
        Capabilities c = null;//super.getCapabilities();
        
        return c;
    }
}
