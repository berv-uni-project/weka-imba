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
    
    public String filter;
    
    public boolean wasNumeric;
    
    public NBTubes() {
        dataClassifier = new ArrayList<>();
        infoClassifier = new ArrayList<>();
        dataset = null;
        sumClass = null;
        filter = "NumericToNominal";
        dataSize = 0;
        wasNumeric = false;
    }
    
    public NBTubes(String filterChoice) {
        dataClassifier = new ArrayList<>();
        infoClassifier = new ArrayList<>();
        dataset = null;
        sumClass = null;
        filter = filterChoice;
        dataSize = 0;
        wasNumeric = false;
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
        
        Filter f;
        int i, j, k, l, m;
        int sumVal;
        
        int numAttr = data.numAttributes(); //ini beserta kelasnya, jadi atribut + 1
        
        i = 0;
        while (i < numAttr && wasNumeric == false) {
            if (i == classIdx) {
                i++;
            }
            
            if (i != numAttr && data.attribute(i).isNumeric()) {
                wasNumeric = true;
            }
            
            i++;
        }

        Instance p;        
        
        //kasih filter
        if (wasNumeric) {
            f = new Normalize();
            //Filter f = new NumericToNominal();
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

            while ((p = f.output()) != null) {
                dataset.add(p);
            }
        }
        
        //f = new NumericToNominal();
        if (filter.equals("Discretize")) {
            f = new Discretize();
        } else {
            f = new NumericToNominal();
        }
        
        try {
            if (wasNumeric) {
                f.setInputFormat(dataset);
                for (Instance i1 : dataset) {
                    f.input(i1);
                }
            } else {
                f.setInputFormat(data);
                for (Instance i1 : data) {
                    f.input(i1);
                }
            }
            
            f.batchFinished();
        } catch (Exception ex) {
            Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        dataset = null;
        dataset = f.getOutputFormat();
        
        while ((p = f.output()) != null) {
            dataset.add(p);
        }
        
        //building data structure
        classIdx = data.classIndex();
        
        dataSize = data.size();
        
        //isi data dan info classifier dengan array kosong
        i = 0;
        j = i;
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
        while (i < dataset.size()) {
            j = 0;
            k = j;
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
        Instances temp = null;
        Instance p;
        Filter f;
        double[] a = new double[infoClassifier.get(0).get(0).size()];
        int i, j, k, l, x, c;
        double t, prev;
        Enumeration n;
        boolean big;
        String val;
        String[] valMinMax;
        
        if (wasNumeric) {
            
            header_Instances.add(instance);
            
            f = new Normalize();
            try {
                f.setInputFormat(header_Instances);

                for (Instance i1 : header_Instances) {
                    f.input(i1);
                }

                f.batchFinished();
            } catch (Exception ex) {
                Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
            }

            temp = f.getOutputFormat();

            while ((p = f.output()) != null) {
                temp.add(p);
            }
        }
        
        f = new NumericToNominal();
            
        if (wasNumeric) {
            try {
                f.setInputFormat(temp);
                

                for (Instance i1 : temp) {
                    f.input(i1);
                }

                f.batchFinished();
            } catch (Exception ex) {
                Logger.getLogger(NBTubes.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            temp = null;
            temp = f.getOutputFormat();

            p = null;

            while ((p = f.output()) != null) {
                temp.add(p);
            }

            instance = temp.lastInstance();
            
            header_Instances.remove(header_Instances.size()-1);
        } else {
            f.setInputFormat(header_Instances);
            
            f.input(instance);
            
            f.batchFinished();
            
            instance = f.output();
        }
                
        //Itung distribusi instance utk tiap kelas
        i = 0;
        while (i < (a.length)) {
            a[i] = (double) sumClass[i] / dataSize;
            
            j = 0;
            k = 0;
            while (j < infoClassifier.size()) {
                
                if (j == classIdx) {
                    k++;
                }
                
                if (wasNumeric) {
                    if (filter.equals("Discretize")) {
                        l = 0;
                        big = false;
                        while (l < dataset.attribute(k).numValues() && big == false) {
                            //parse
                            val = String.valueOf(dataset.attribute(k).value(l));
                            //System.out.println("k = " + k);
                            //System.out.println("nilai = " + instance.stringValue(k));
                            val = val.replaceAll("'", "");
                            val = val.replaceAll("\\(", "");
                            val = val.replaceAll("\\)", "");
                            val = val.replaceAll("]", "");
                            
                            valMinMax = val.split("-");
                            
                            
                            //cocokin
                            
                            
                            if (valMinMax.length == 3) {
                                if (valMinMax[1].equals("inf")) {
                                    valMinMax[1] = "0.0";
                                }
                                //System.out.println("Min = " + valMinMax[1]);
                                //System.out.println("Max = " + valMinMax[2]);
                                if (Double.valueOf(instance.stringValue(k)) > Double.valueOf(valMinMax[1]) && Double.valueOf(instance.stringValue(k)) <= Double.valueOf(valMinMax[2])) {
                                    big = true;
                                }
                            } else {
                                if (valMinMax[1].equals("inf")) {
                                    valMinMax[1] = "1.0";
                                }
                                //System.out.println("Min = " + valMinMax[0]);
                                //System.out.println("Max = " + valMinMax[1]);
                                if (Double.valueOf(instance.stringValue(k)) > Double.valueOf(valMinMax[0]) && Double.valueOf(instance.stringValue(k)) <= Double.valueOf(valMinMax[1])) {
                                    big = true;
                                }
                            }
                            l++;
                        }
                        
                        x = l - 1;
                
                        //System.out.println("x = " + x);
                    } else {
                        big = false;
                        l = 0;
                        n = dataset.attribute(k).enumerateValues();

                        t = 0;
                        prev = 0;
                        while (l < dataset.attribute(k).numValues() && big == false) {
                            t = Double.valueOf(n.nextElement().toString());

                            //System.out.println(prev + "   " + t);
                            if (Double.valueOf(instance.stringValue(k)) <= t) {
                                big = true;
                            } else {
                                prev = t;
                            }

                            l++;    
                        }

                        if (big == true && t != Double.valueOf(instance.stringValue(k))) {
                            System.out.println(prev + "   " + Double.valueOf(instance.stringValue(k)) + "   " + t);
                        }
                        //x = l - 1;

                        if (classIdx < 2) {
                            c = 2;
                        } else {
                            c = 1;
                        }

                        if (big == true && l > c) {
                            if ((Double.valueOf(instance.stringValue(k)) - prev) <= (t - Double.valueOf(instance.stringValue(k)))) {
                                x = l - 2;
                            } else {
                                x = l - 1;
                            }
                        } else {
                            x = l - 1;
                        }
                    }
                    
                } else {
                    x = dataset.attribute(k).indexOfValue(instance.stringValue(k));
                }
                
                a[i] *= infoClassifier.get(j).get(x).get(i);
                                    
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