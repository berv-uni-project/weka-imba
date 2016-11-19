/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imba.classifier;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Utils;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author absol
 */
public class NBTubes extends AbstractClassifier {
    
    /** for serialization */
  static final long serialVersionUID = 5995231201785697655L;
    
    public ArrayList<ArrayList<ArrayList<Integer>>> dataClassifier;
    public ArrayList<ArrayList<ArrayList<Double>>> infoClassifier;
    protected Instances dataset;
    protected int numClasses;
    //Urutan: 1. Atribut, 2. Domain, 3. Kelas
    //Kelas dan domain beserta jumlah instance nya dijumlah dari setiap data
    //domain dari sebuah atribut
	
    public int[] sumClass;
    public int dataSize;
    public int classIdx;
    
     /** The attribute estimators. */
    protected Estimator[][] m_Distributions;

    /** The class estimator. */
    protected Estimator m_ClassDistribution;
    
    public NBTubes() {
        dataClassifier = new ArrayList<>();
        infoClassifier = new ArrayList<>();
        dataset = null;
        sumClass = null;
        dataSize = 0;
    }
	
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        getCapabilities().testWithFail(data);
        
        // hapus data dengan kelas yang hilang
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        // copy data
        dataset = new Instances(data);
		
        
        int i, j, k, l;
        int sumVal = 0;
        
        int numAttr = dataset.numAttributes();
/*        
        NumericToNominal filter = new NumericToNominal();
        filter.setInputFormat(this.dataset);
        this.dataset = Filter.useFilter(this.dataset, filter);
*/        
        // discretize
        Discretize discret = new Discretize();
        discret.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset,discret);
        
        // Reserve space for the distributions
        m_Distributions = new Estimator[dataset.numAttributes() - 1][dataset.numClasses()];
        m_ClassDistribution = new DiscreteEstimator(dataset.numClasses(), true);
        int attIndex = 0;
        Enumeration<Attribute> enu = dataset.enumerateAttributes();
        while (enu.hasMoreElements()) {
            Attribute attribute = enu.nextElement();
            for (int m = 0; m < dataset.numClasses(); m++) {
                m_Distributions[attIndex][m] = new DiscreteEstimator(attribute.numValues(), true);
            }
            attIndex++;
        }
        
        // Compute counts
        Enumeration<Instance> enumInsts = dataset.enumerateInstances();
        while (enumInsts.hasMoreElements()) {
          Instance instance = enumInsts.nextElement();
          updateClassifier(instance);
        }

        // Save space
        dataset = new Instances(dataset, 0);
        
        /*
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
        } */
    }
    
     /**
   * Updates the classifier with the given instance.
   * 
   * @param instance the new training instance to include in the model
   * @exception Exception if the instance could not be incorporated in the
   *              model.
   */
  public void updateClassifier(Instance instance) throws Exception {

    if (!instance.classIsMissing()) {
      Enumeration<Attribute> enumAtts = dataset.enumerateAttributes();
      int attIndex = 0;
      while (enumAtts.hasMoreElements()) {
        Attribute attribute = enumAtts.nextElement();
        if (!instance.isMissing(attribute)) {
          m_Distributions[attIndex][(int) instance.classValue()].addValue(
            instance.value(attribute), instance.weight());
        }
        attIndex++;
      }
      m_ClassDistribution.addValue(instance.classValue(), instance.weight());
    }
  }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] probs = new double[numClasses];
        for (int j = 0; j < numClasses; j++) {
          probs[j] = m_ClassDistribution.getProbability(j);
        }
        Enumeration<Attribute> enumAtts = instance.enumerateAttributes();
        int attIndex = 0;
        while (enumAtts.hasMoreElements()) {
            Attribute attribute = enumAtts.nextElement();
            if (!instance.isMissing(attribute)) {
                double temp, max = 0;
                for (int j = 0; j < numClasses; j++) {
                    temp = Math.max(1e-75, Math.pow(m_Distributions[attIndex][j]
                      .getProbability(instance.value(attribute)),
                      dataset.attribute(attIndex).weight()));
                    probs[j] *= temp;
                    if (probs[j] > max) {
                        max = probs[j];
                    }
                    if (Double.isNaN(probs[j])) {
                        throw new Exception("NaN returned from estimator for attribute "
                        + attribute.name() + ":\n"
                        + m_Distributions[attIndex][j].toString());
                    }
                }
                if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
                  for (int j = 0; j < numClasses; j++) {
                    probs[j] *= 1e75;
                  }
                }
            }
            attIndex++;
        }

        return probs;
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
<<<<<<< HEAD
}
=======
    
    /**
   * Main method for testing this class.
   * 
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new NBTubes(), argv);
  }
}
>>>>>>> c6ec05658ab7ddce96a7dc2f2c5dab93e95f4cdc
