/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.util.ArrayList;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 *
 * @author absol
 */
public class NBTubes extends AbstractClassifier {
    
    public ArrayList<ArrayList<ArrayList<Integer>>> dataClassifier;
    public ArrayList<ArrayList<ArrayList<Double>>> infoClassifier;
    public Instances dataset;
    protected Instances header_Instances;
    //Urutan: 1. Atribut, 2. Domain, 3. Kelas
    //Kelas dan domain beserta jumlah instance nya dijumlah dari setiap data
    //domain dari sebuah atribut
	
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
        buildClassifier(data, 0);
    }
    
    public void buildClassifier(Instances data, int cI) {
		// test data
        getCapabilities().testWithFail(data);
        
        // hapus data dengan kelas yang hilang
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        // copy data
        header_Instances = new Instances(data);
		
		int i, j, k, l;
        int sumVal = 0;
        
        int numAttr = data.get(0).numAttributes();
        //int numClasses = data.get(0).attribute(numAttr-1).numValues();
        int numClasses = data.get(0).attribute(classIdx).numValues();
        
        //kasih filter
        Filter f = new NumericToNominal();
        try {
            f.setInputFormat(data);
            
            for (Instance i1 : data) {
                f.input(i1);
            }
            
            f.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        Instances filteredData = f.getOutputFormat();
        
        Instance p;

        while ((p = f.output()) != null) {
            filteredData.add(p);
        }
        
        //masukin dataset dari filteredData
        dataset = filteredData;
        
        //building data structure
        sumClass = new int[numClasses];
        dataClassifier = new ArrayList<>();
        dataSize = filteredData.size();
        classIdx = cI;
        
        int a = 0;
        i = a;
        while (a < numAttr) {
            if (a != classIdx) {
                //add new attribute, blsm ada domainnya
                dataClassifier.add(new ArrayList<>());
                infoClassifier.add(new ArrayList<>());
            
                j = 0;
                while (j < filteredData.get(0).attribute(i).numValues()) {
                    //add new domain pada suatu atribut, blm ada perbedaan kelas
                    dataClassifier.get(i).add(new ArrayList<>());
                    infoClassifier.get(i).add(new ArrayList<>());

                    k = 0;
                    while (k < filteredData.get(0).attribute(classIdx).numValues()) {
                        //add new kelas di dalam atribut, domain spesifik                    
                        dataClassifier.get(i).get(j).add(0);
                        infoClassifier.get(i).get(j).add(0.0);
                        k++;
                    }   
                    j++;
                }
            } else {
                i--;
            }
               
            a++;
            i++;
        }
        
        System.out.println("classId = " + classIdx);
        
        //reading from instances
        i = 0;
        while (i < filteredData.size()) {
            a = 0;
            j = a;
            while (a < numAttr) {
                if  (a != classIdx) {
                    dataClassifier.get(j).
                            get((int) filteredData.
                                        get(i).
                                            value(j)).
                            set((int) filteredData.get(i).value(classIdx), 
                                    dataClassifier.get(j).get(
                                            (int) filteredData.get(i).value(j)).get(
                                                    (int) filteredData.get(i).value(classIdx))+1);

                    if (j == 0) {
                        sumClass[(int) filteredData.
                                get(i).
                                value(classIdx)]++;
                    }
                }
                else {
                    j--;
                }
                
                j++;
                a++;
            }
            
            i++;
        }
        
        //getting double values, jumlahNilaiXdiAtrY/jumlahAtrYDiKelasZ
        i = 0;
        while (i < dataClassifier.size())
        {
            j = 0;
            while (j < dataClassifier.get(i).size()) {
                k = 0;
                while (k < dataClassifier.get(i).get(j).size()) {
                    //dapetin sumVal(j) utk kelas(k)
                    sumVal = dataClassifier.get(i).get(j).get(k);                    

                    infoClassifier.get(i).
                            get(j).
                            set(k, (double)sumVal/sumClass[k]);
                    k++;
                }
                j++;
            }
            i++;
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //Fungsi ini menentukan probabilitas setiap kelas instance untuk instance 
        //yang ada di parameter fungsi
        
        //kasih filter
        Filter f = new NumericToNominal();
        
        f.setInputFormat(dataset);
        
        f.input(instance);
        
        f.batchFinished();
        
        instance = f.output();
        
        //Classify~
        double[] a = new double[infoClassifier.get(0).get(0).size()];
        
        int i = 0;
        int j = 0;
        int k = 0;
        while (i < (a.length)) {
            a[i] = (double) sumClass[i] / dataSize;
            
            System.out.println("infoClassifier.size() = " + infoClassifier.size());
            
            j = 0;
            k = 0;
            while (j < infoClassifier.size()) {
                    a[i] *= infoClassifier.get(j).
                            get((int)instance.
                                    value(k)). //salah di sini~
                                get(i);

                    System.out.println("j = " + j);
                    System.out.println("k = " + k);
                
                k++;
                j++;
            }
            
            i++;
        }
        
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
