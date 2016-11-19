/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author absol
 */
public class NBTubes extends AbstractClassifier implements Serializable {
    
    //Urutan: 1. Atribut, 2. Domain, 3. Kelas
    //Kelas dan domain beserta jumlah instance nya dijumlah dari setiap data
    //domain dari sebuah atribut
    public ArrayList<ArrayList<ArrayList<Integer>>> dataClassifier;
    public ArrayList<ArrayList<ArrayList<Double>>> infoClassifier;
    
    public ArrayList<ArrayList<Boolean>> validAttribute;
    
    public Instances dataset;
    protected Instances header_Instances;
	
    public int[] sumClass;
    public int dataSize;
    public int classIdx;
    
    public NBTubes() {
        dataClassifier = new ArrayList<>();
        infoClassifier = new ArrayList<>();
        dataset = null;
        sumClass = null;
        dataSize = 0;
    }
	
    @Override
    public void buildClassifier(Instances data) {
        dataClassifier = new ArrayList<>();
        infoClassifier = new ArrayList<>();
        validAttribute = new ArrayList<>();
        dataset = null;
        sumClass = null;
        dataSize = 0;
        header_Instances = data;
        
        //kasih filter
        Filter f = new Normalize();
        try {
            f.setInputFormat(data);
            
            for (Instance i1 : data) {
                f.input(i1);
            }
            
            f.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        dataset = f.getOutputFormat();
        
        Instance p = null;

        while ((p = f.output()) != null) {
            dataset.add(p);
        }
        
        f = new NumericToNominal();
        try {
            f.setInputFormat(dataset);
            
            for (Instance i1 : dataset) {
                f.input(i1);
            }
            
            f.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        dataset = null;
        dataset = f.getOutputFormat();
        
        p = null;

        while ((p = f.output()) != null) {
            dataset.add(p);
        }
        
        //int x, y, z;
        
        /*
        Enumeration n;
        for (x = 0; x < dataset.numAttributes(); x++) {
            n = dataset.attribute(x).enumerateValues();
            for (y = 0; y < dataset.attribute(x).numValues(); y++) {
                System.out.print(n.nextElement() + "   ");
            }
            System.out.println();
        }
        */
        
        //building data structure
        int i, j, k, l, m;
        int sumVal;
        
        int numAttr = data.numAttributes(); //ini beserta kelasnya, jadi atribut + 1
        
        classIdx = data.classIndex();
        
        System.out.println(classIdx);
        
        dataSize = data.size();
        
        //isi data dan info classifier dengan array kosong
        i = 0;
        j = i;
        m = j;
        while (j < numAttr) {
            if (i == classIdx) {
                i++;
            } else {
                dataClassifier.add(new ArrayList<>());
                infoClassifier.add(new ArrayList<>());
                
                if (j < i) {
                    m = j - 1;
                } else {
                    m = j;
                }
                
                k = 0;
                while (k < dataset.attribute(j).numValues()) {
                    dataClassifier.get(m).add(new ArrayList<>());
                    infoClassifier.get(m).add(new ArrayList<>());                
                    
                    l = 0;
                    while (l < dataset.attribute(classIdx).numValues()) {
                        dataClassifier.get(m).get(k).add(0);
                        infoClassifier.get(m).get(k).add(0.0);
                        
                        l++;
                    }
                    
                    k++;
                }
            }
            
            i++;
            j++;
        }
        
        //isi data classifier dari dataset
        sumClass = new int[data.numClasses()];
        
        i = 0;
        m = 0;
        while (i < dataset.size()) {
            j = 0;
            k = j;
            m = k;
            while (k < dataset.numAttributes()) {
                if (j == classIdx) {
                    j++;
                } else {
                    if (k < j) {
                        m = k - 1;
                    } else {
                        m = k;
                    }
                    
                    dataClassifier.get(m).get((int)dataset.get(i).value(k)).set(
                        (int)dataset.get(i).value(classIdx),
                            dataClassifier.get(m).get((int)dataset.get(i).value(k)).get((int)dataset.get(i).value(classIdx))+1);
                    
                    if (m == 0) {
                        sumClass[(int)dataset.get(i).value(classIdx)]++;
                    }
                    
                }
                
                k++;
                j++;
            }
            
            i++;
        }
        
        //proses double values
        i = 0;
        while (i < dataClassifier.size()) {
            j = 0;
            while (j < dataClassifier.get(i).size()) {
                k = 0;
                while (k < dataClassifier.get(i).get(j).size()) {
                    infoClassifier.get(i).
                            get(j).set(k, (double)dataClassifier.get(i).get(j).get(k)/sumClass[k]);
                    
                    k++;
                }
                
                j++;
            }
            
            i++;
        }
        
        /*
        //liat apakah ada nilai di tiap atribut
        //yang merepresentasikan lebih dari 80% data
        i = 0;
        while (i < dataClassifier.size()) {
            j = 0;
            while (j < dataClassifier.get(i).size()) {
                
                
                j++;
            }
            
            i++;
        }
*/
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //Fungsi ini menentukan probabilitas setiap kelas instance untuk instance 
        //yang ada di parameter fungsi
        //System.out.println(instance.toString());
        
        System.out.println(instance.attribute(0).isNominal());
        System.out.println(instance.attribute(0).isNumeric());
                
        //kasih filter
        /*
        Filter f = new Normalize();
        
        f.setInputFormat(header_Instances);
        
        f.input(instance);
        
        f.batchFinished();
        
        Instances temp;
        temp = f.getOutputFormat();
        temp.add(f.output());
        */
        //System.out.println("temp = " + temp.toString());
        
        Filter f = new NumericToNominal();
        
        f.setInputFormat(header_Instances);
        
        f.input(instance);
        
        f.batchFinished();
        
        instance = f.output();
        
        System.out.println("baru = " + instance.toString());
        
        
        /*
        Enumeration n;
        n = dataset.attribute(1).enumerateValues();
        
        
        while (n.hasMoreElements()) {
            System.out.println(n.nextElement() +  "   ");
        }
        */
        
        //Classify~
        double[] a = new double[infoClassifier.get(0).get(0).size()];
                
        int i = 0;
        int j;
        int k;
        int x;
        while (i < (a.length)) {
            a[i] = (double) sumClass[i] / dataSize;
            
            //System.out.println("prob kelas " + i + " = " + a[i]);
            
            j = 0;
            k = 0;
            while (j < infoClassifier.size()) {
                
                if (j == classIdx) {
                    k++;
                }
                
                //System.out.println(dataset.attribute(j).isNominal());
                //System.out.println(dataset.attribute(j).isNumeric());
                
                /*
                if (instance.value(k) % 1 == 0) {
                    System.out.println("shit yes");
                    //x = dataset.attribute(j).indexOfValue(String.valueOf((int)instance.value(k)));
                    x = dataset.attribute(j).indexOfValue(instance.stringValue(k) + ".0");
                } else {
                    System.out.println("thank god no");
                    x = dataset.attribute(j).indexOfValue(instance.stringValue(k));
                }
                */

                x = dataset.attribute(k).indexOfValue(instance.stringValue(k));
                //System.out.println("k = " + k);
                //System.out.println("indeks = " + dataset.attribute(k).indexOfValue(instance.stringValue(k)));
                //System.out.print("prob kelas " + i + " given value " + instance.stringValue(k) + "(indeks ke-" + x + ") pada atribut ke " + j + " = " + " ");
                //System.out.println(dataClassifier.get(j).get(x).get(i) + "  " + infoClassifier.get(j).get(x).get(i));
                
                a[i] *= infoClassifier.get(j).get(x).get(i);
                
                //dataset.attribute(j).indexOfValue(String.valueOf((int)p.value(k)));
                
                    
                k++;
                j++;
            }
            
            //System.out.println("prob kelas " + i + " final = " + a[i]);
            //System.out.println();
            
            i++;
        }
        
        //System.out.println();
        
        return a;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //Fungsi ini mengembalikan indeks kelas dengan probabilitas tertinggi
        
        //panggil distributionForInstance
		double[] a = distributionForInstance(instance);
        
        //cari max value, return indeks kelas paling tinggi wqwqw
		double max = 0.0;
		int maxIdx = 0;
		int i = 0;
		while (i < a.length) {
			if (a[i] > max) {
				max = a[i];
				maxIdx = i;
			}
			
			i++;
		}
        
        return maxIdx;
    }
    
    @Override
    public Capabilities getCapabilities() {
        //Fungsi ini mengembalikan "Capabilities", yaitu handler kasus2 aneh pada
        
        Capabilities c = super.getCapabilities();
        c.disableAll();
        
        // attributes
        c.enable(Capability.NOMINAL_ATTRIBUTES);
        c.enable(Capability.NUMERIC_ATTRIBUTES);
        c.enable(Capability.MISSING_VALUES);
        
        // class
        c.enable(Capability.NOMINAL_CLASS);
        c.enable(Capability.MISSING_CLASS_VALUES);
        
        // instances
        c.setMinimumNumberInstances(0);
        
        return c;
    }
}
