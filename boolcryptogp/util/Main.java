import java.math.BigInteger;
import java.util.Vector;
import java.util.Random;
import java.util.Arrays;



public class Main {
    
    public static int[] generateRandomTruthTable(Random r, int n) {
        int space_cardinality = (int)Math.pow(2, n);
        int[] v = new int[space_cardinality];
        for (int i = 0; i < space_cardinality; i++){
            v[i] = r.nextInt(2);
        }
        return v;
    }
    
    public static int[][] generateKRandomTruthTables(Random r, int n, int k) {
        int space_cardinality = (int)Math.pow(2, n);
        int[][] v = new int[k][space_cardinality];
        for (int i = 0; i < k; i++){
            v[i] = generateRandomTruthTable(r, n);
        }
        return v;
    }
    

    
    /**
     * Computes the Walsh Transform of a Boolean function using the Fast Walsh
     * Transform (FWT) algorithm, which requires O(NlogN) operations (N=2^n is
     * the length of the truth table). The method directly computes the spectrum
     * in the original vector, and it must be called with the initial parameters
     * (vector, 0, vector.length). The return value is the spectral radius of
     * the function (=maximum absolute value of the Walsh transform).
     * The reference for the algorithm is C. Carlet, "Cryptography and
     * Error-Correcting Codes", chapter 8 in Crama, Hammer, "Boolean Models and
     * Methods in Mathematics, Computer Science and Engineering", p. 272.
     * 
     * @param vector    An array of int representing the truth table of a
     *                  Boolean function in polar form.
     * @param start     The index of the truth table where to start computations.
     * @param length    Length of the portion of truth table where to perform
     *                  computations, starting from start
     * @return          The spectral radius of the function, computed as the
     *                  maximum absolute value of the Walsh transform
     */
    public static int computeFWT(int[] vector, int start, int length) {
        
        int half = length/2;
        
        //Main cycle: split vector in two parts (v0 e v1), 
        //update v0 as v0=v0+v1, and v1 as v1=v0-v1.
        for(int i=start; i<start+half; i++) {
            
            int temp = vector[i];
            vector[i] += vector[i+half];
            vector[i+half] = temp - vector[i+half];
            
        }
        
        //Recursive call on v0 and v1.
        if(half>1) {
            
            int val1 = computeFWT(vector,start,half);
            int val2 = computeFWT(vector,start+half,half);
            
            //At the end of the recursive calls, compare val1 and val2 to decide
            //what is the spectral radius in the portion of truth table included
            //between start and start+length
            if(val1 > val2) {
                    
                return val1;
                
            }
            else {
                
                return val2;
                
            }

        } else {
        
            //If we have reached half=1 (function of 2 variables),
            //return the highest coefficient in absolute value.
            if(Math.abs(vector[start]) > Math.abs(vector[start+half]))
                return Math.abs(vector[start]);
            else
                return Math.abs(vector[start+half]);           
            
        }
        
    }
    
    /**
     * Computes the inverse Walsh Transform using the Fast Walsh Transform (FWT)
     * algorithm, which requires O(NlogN) operations (N=2^n is the length of the
     * truth table). Starting from the Walsh transform vector of a Boolean
     * function, the method returns its truth table in polar form. The method
     * directly computes the truth table in the original vector, and
     * it must be called with the initial parameters (vector, 0, vector.length);
     * 
     * @param vector    an array of integers representing the Walsh transform of a
     *                  Boolean function.
     * @param start     the index of the Walsh transform where to start computations.
     * @param length    length of the portion of Walsh transform where to perform
     *                  computations, starting from start.
     * @return          The maximum absolute value in the output vector. This
     *                  can be used as the maximum autocorrelation coefficient
     *                  when computing the autocorrelation function via the
     *                  Wiener-Khintchine theorem (i.e., starting from the
     *                  (squared Walsh transform and applying the inverse Walsh
     *                  transform).
     * 
     */
    public static int computeInvFWT(int[] vector, int start, int length) {
        
        int half = length/2;
        
        //Main cycle: split vector in two parts (v0 e v1), 
        //update v0 as v0=(v0+v1)/2, and v1 as v1=(v0-v1)/2.
        //Division by 2 is a normalization factor.
        for(int i=start; i<start+half; i++) {
            int temp = vector[i];
            vector[i] = (int)((vector[i] + vector[i+half]) / 2);
            vector[i+half] = (int)((temp - vector[i+half]) / 2);
        }
        
        //Recursive call on v0 and v1.
        if(half>1) {
            
            int val1 = computeInvFWT(vector,start,half);
            int val2 = computeInvFWT(vector,start+half,half);
            
            //At the end of the recursive calls, compare val1 and val2 to decide
            //what is the spectral radius in the portion of truth table included
            //between start and start+length
            if(val1 > val2) {
                    
                    return val1;
                    
            }
            
            else {
                
                return val2;
                
            }

        } else {
        
            //If we have reached half=1 (function of 2 variables),
            //return the highest coefficient in absolute value
            
            //If we are checking the null position, return the index in the
            //first position. This is because this method is used to calculate
            //the autocorrelation function, whose value ACmax must be computed
            //only on the non-zero vectors.
            if(start == 0) {
                return Math.abs(vector[1]);
            } else {
                
                if(Math.abs(vector[start]) > Math.abs(vector[start+half])) {
                    return Math.abs(vector[start]);
                }
                else {
                    return Math.abs(vector[start+half]);           
                }
                
            }
        }
        
    }

    
    /**
     * Computes the autocorrelation function of a boolean function, using the
     * Wiener-Khintchine theorem. The general computation flow is the following:
     * 
     * 1) Given the truth table of f, compute the Fast Walsh Transform (FWT) of f
     * 2) Compute the square of the Walsh transform of f
     * 3) Apply the Inverse Walsh Transform to the squared Walsh transform of f.
     *    The result is the autocorrelation function of f.
     * 
     * If a boolean flag is set, this method performs all three steps above,
     * and assumes that the input vector holds the polar form of the truth table
     * of the function. Otherwise, only step 2 and 3 are performed, and it is
     * assumed that the input vector contains the Walsh transform of the
     * function. This flag can be used to save computations if the Walsh
     * transform has already been computed in advance.
     * 
     * Further, the method directly computes the autocorrelation function in the
     * original input vector, and it returns the maximum autocorrelation coefficient.
     * 
     * @param vector    An array of int representing either the polar form of
     *                  the truth table of the Boolean function or its Walsh
     *                  Transform, depending on the mode flag
     * @param mode      a Boolean flag specifying whether vector is the polar
     *                  truth table of the function (true) or the Walsh spectrum
     *                  (false), used to avoid unnecessary computations.
     * @return          The maximum autocorrelation coefficient of the function.
     */
    public static int computeAC(int[] vector, boolean mode) {
        
        //(Step 1): Compute the Walsh spectrum of the function, if vector
        //represents the truth table of the function.
        if(mode) {
            
            computeFWT(vector, 0, vector.length);
            
        }
        
        //Step 2: Square the Walsh spectrum
        for(int i=0; i<vector.length; i++) {
            vector[i] *= vector[i];
        }
        
        //Step 3: Compute the inverse WT of the squared spectrum
        //and return the maximum autocorrelation coefficient
        int acmax = computeInvFWT(vector, 0, vector.length);
        return acmax;
        
    }
    
    /**
     * Finds the maximum absolute values of a vector in the positions with
     * specified Hamming weights. Can be used to determine the correlation
     * immunity order t of a Boolean function (by giving in input the Walsh
     * spectrum) or its propagation criterion PC(l) (by giving in input the
     * autocorrelation function).
     * 
     * @param coeffs    An array of coefficients (it can be the Walsh spectrum
     *                  or the autocorrelation function of a Boolean function).
     * @param indices   An array containing arrays of indices whose binary 
     *                  representations have a specified Hamming weight (for
     *                  example, indices[i] contains all indices of Hamming
     *                  weight i). The starting weight is 1, so indices[0]
     *                  must contain the indices of weight 1.
     * @return          An array containing the maximum absolute values in coeffs
     *                  at the positions specified by indices.
     */
    public static int[] computeDevs(int[] coeffs, int[][] indices) {
        
        int[] devs = new int[indices.length];
        
        //Cycle through Hamming weights from 1 to indices.length
        for(int i=0; i<indices.length; i++) {
            
            //Cycle through inputs with Hamming weight i
            for(int j=0; j<indices[i].length; j++) {
                
                int absval = Math.abs(coeffs[indices[i][j]]);
                
                if(absval > devs[i]) {
                    devs[i] = absval;
                }
                
            }
            
            //Deviations of order i must take into account also deviations
            //of lower orders.
            for(int k=0; k<i; k++) {
                
                if(devs[k] > devs[i]) {
                    
                    devs[i] = devs[k];
                    
                }
                
            }
            
        }
        
        return devs;
        
    }
    
    /**
     * Computes the Moebius Transform of a Boolean function using the Fast
     * Moebius Transform (FMT) algorithm, which requires O(NlogN) operations
     * (N=2^n is the length of the truth table). The method directly computes
     * the spectrum in the original vector, and it must be called with the
     * initial parameters (vector, 0, vector.length). The return value of the
     * method is the algebraic degree of the function.
     * 
     * The reference for the algorithm is Carlet, "Cryptography and
     * Error-Correcting Codes", chapter 8 in Crama, Hammer, "Boolean Models and
     * Methods in Mathematics, Computer Science and Engineering", p. 263.
     * 
     * NOTICE: the Moebius transform is an involution, meaning that this method
     * can also be used to recover the truth table of a function starting from
     * its ANF vector.
     * 
     * @param vector    An array boolean representing the boolean function.
     * @param start     The index of the truth table where to start computations.
     * @param length    The length of the subvector where to perform computations
     *                  starting from start.
     * @return          The algebraic degree
     */
    public static int computeFMT(boolean[] vector, int start, int length) {
        
        int half = length/2;
        
        //Main cycle: split vector in two parts (v0 e v1),
        //update v1 as v1=v0 XOR v1.
        for(int i=start; i<start+half; i++) {
            vector[i+half] = vector[i]^vector[i+half];
        }
        
        //Recursive call on v0 and v1.
        if(half>1) {
            
            int val1 = computeFMT(vector,start,half);
            int val2 = computeFMT(vector,start+half,half);
            
            //At the end of the recursive calls, compare val1 and val2 to decide
            //what is the algebraic degree in the portion of ANF included
            //between start and start+length
            if(val1 > val2) {
                    
                    return val1;
                    
            }
            else {
                
                return val2;
                
            }

        } else {
        
            //If we have reached half=1 (function of 2 variables),
            //return the Hamming weight of the largest input with nonzero
            //coefficient in the current subvector. This is the algebraic degree
            //of the restriction of the function of 2 variables.
                
            //If both coefficient are zero, then the degree of the subfunction
            //is zero.
            if((vector[start] == false) && (vector[start+half] == false)) {
                
                return 0;
                
            } else {

                //Compute the length of the input vectors.
                int inplen = (int)(Math.log(vector.length)/Math.log(2));

                //If the coefficient of the higher vector is null,
                //then return the Hamming weight of the lower vector.
                if(vector[start+half] == false) {

                    boolean[] input = dec2Bin(start, inplen);
                    int subdeg = hammingWeight(input);
                    
                    return subdeg;

                } else {

                    //In all other cases, return the Hamming weight of the
                    //higher vector.
                    boolean[] input = dec2Bin(start+half, inplen);
                    int subdeg = hammingWeight(input);
                    
                    return subdeg;

                }
            }
            
        }
        
    }
    
    /**
     * Computes the algebraic expression of the ANF of a Boolean function,
     * represented as a multivariate polynomial.
     * 
     * @param anfcoeffs A boolean array holding the ANF coefficients of the function
     * @param nvar      Number of variables of the function
     * @return          A string representing the algebraic expression of the
     *                  ANF of the function, as a multivariate polynomial.
     */
    public static String computeANFExpr(boolean[] anfcoeffs, int nvar) {
        
        //Find the last nonzero coefficient in the ANF
        int last = 0;
        int k = anfcoeffs.length-1;
        
        while((last == 0) && (k>=0)) {
            
            if(anfcoeffs[k])
                last = k;
            
            k--;
            
        }
        
        //Initialize the ANF string
        String anfexpr = "f(";

        for(int i=0; i<nvar; i++) {
            anfexpr += "x"+(i+1);
            if(i<nvar-1) {
                anfexpr += ",";
            }
        }
        anfexpr += ") = ";
        
        if(anfcoeffs[0]) {
            anfexpr += "1 + ";
        }
        
        for(int i=1; i<=last; i++) {
            
            if(anfcoeffs[i]) {
                
                //Print the i-th variation of variables
                boolean[] input = dec2Bin(i,nvar);
                
                for(int j=0; j<nvar; j++) {
                    
                    if(input[j]) {
                        
                        anfexpr += "x"+(j+1);
                        
                    }
                    
                }
                
                if(i<last) {
                    anfexpr += " + ";
                }
                
                
            }
            
        }
        
        return anfexpr;
        
    }
    
    /**
    * Compute the bitwise XOR between two boolean arrays of equal length.
    * 
    * @param seq1   a boolean array (first operand)
    * @param seq2   a boolean array (second operand)
    * @return       the bitwise XOR of seq1 and seq2
    */
   public static boolean[] xorBits(boolean[] seq1, boolean[] seq2) {

       boolean[] xoredSeq = new boolean[seq1.length];

       for(int i=0; i<seq1.length; i++) {
           xoredSeq[i] = seq1[i] ^ seq2[i];
       }

       return xoredSeq;
       
   }
   
   /**
     * Computes the Hamming weight of a boolean array (=number of positions
     * equal to true).
     * 
     * @param binStr    The boolean vector whose Hamming weight has to be
     *                  computed.
     * @return          The Hamming weight of binStr.
     */
    public static int hammingWeight(boolean[] binStr) {
        
        int weight = 0;
        
        for(int i=0; i<binStr.length; i++) {
            
            if(binStr[i])
                weight++;
            
        }
        
        return weight;
        
    }
    
    /**
     * Computes the Hamming distance between two boolean arrays of equal length
     * (= number of positions in which the two arrays differ).
     * 
     * @param binStr1   First boolean array
     * @param binStr2   Second boolean array
     * @return          The Hamming distance between binStr1 and BinStr2
     */
    public static int hammingDist(boolean[] binStr1, boolean[] binStr2) {
        
        //Compute the bitwise XOR of the two strings
        boolean[] xor = xorBits(binStr1, binStr2);
        
        //Compute the Hamming weight of the XOR, which is equal to the Hamming
        //distance of the two strings
        int hdist = hammingWeight(xor);
        
        return hdist;
        
    }

    /**
     * Computes the 1-complement of a boolean array
     * 
     * 
     * @param   binStr a boolean array
     * @return         the 1-complement of binStr.
     */
    public static boolean[] complement(boolean[] binStr) {
        boolean[] compl = new boolean[binStr.length];

        for(int i=0; i<compl.length; i++) {
            compl[i] = !binStr[i];
        }

        return compl;
    }   
    
    /**
     * Inverts the order of a boolean array.
     * 
     * 
     * @param   binStr  a boolean array
     * @return          the reverse of binStr
     */
    public static boolean[] reverse(boolean[] binStr) {
            
        boolean[] rev = new boolean[binStr.length];

        for(int i=0; i<rev.length; i++) {

            rev[i] = binStr[binStr.length-1-i];

        }

        return rev;

    }
    
    /**
     * Compute the scalar product between two boolean vectors of equal length.
     * If the length of the vectors is n, the scalar product is computed through
     * the following formula:
     * 
     * vect1.vect2 = XOR_{i=1}^{n} (vect1[i] AND vect2[i])
     * 
     * @param vect1     First boolean vector
     * @param vect2     Second boolean vector
     * @return          The scalar product between vect1 and vect2, which is
     *                  equal to the XOR of the bitwise ANDs between vect1 and
     *                  vect2
     */
    public static boolean scalarProduct(boolean[] vect1, boolean[] vect2) {
        
        boolean prod = false;
        
        for(int i=0; i<vect2.length; i++) {
            
            prod ^= vect1[i] && vect2[i];
            
        }
        
        return prod;
        
    }
    
    /**
     * Computes the factorial of an integer number.
     * 
     * @param num   an integer number
     * @return      the factorial of num
     */
    public static int factorial(int num) {
        
        int fact = 1;
        
        if((num==0) || (num==1)) {            
            return fact;   
        }
        
        for(int i=2; i<=num; i++) {
            fact *= i;
        }
        
        return fact;
        
    }
    
    /**
     * Computes the binomial coefficient (n,k).
     * 
     * @param n         the size of the set from which combinations are drawn.
     * @param k         the size of the combinations.
     * @return          the binomial coefficient (n,k)
     */
    public static int binCoeff(int n, int k) {
        
        long numerator = 1;
        
        for(int i=n-k+1; i<=n; i++) {
            numerator *= i;
        }
        
        long denominator = factorial(k);
        int bCoeff = (int)(numerator/denominator);
        
        return bCoeff;
        
    }
    
    /**
     * Generates all (s+t)-bitstrings with Hamming weight t, in decimal
     * notation. The algorithm is described in Knuth, "The Art of Computer
     * Programming, pre-Fascicle 4A" (Algorithm L, p. 4).
     * 
     * @param s         number of 0s in the bitstrings
     * @param t         number of 1s in the bitstrings
     * @return          array of integers representing the bitstrings of length
     *                  (s+t) and Hamming weight t
     */
    public static int[] genBinCombs(int s, int t) {

        int size = binCoeff(s+t, t);
        int[] combset = new int[size];
        
        int index = 0; //index for the set combs.
        
        //Initialisation
        int[] comb = new int[t+2]; //the two additional cells are sentinels.
        for(int j=0; j<t; j++) {
            comb[j] = j;
        }
        comb[t] = s+t;
        comb[t+1] = 0;
        
        int j = 0;
        
        while(j<t) {
            
            boolean[] conf = new boolean[s+t];
            
            for(int k=0; k<t; k++) {
                conf[comb[k]] = true;
            }
            
            //Convert the combination in decimal notation and
            //copy it in the final set.
            int deccomb = bin2DecInt(conf);
            combset[index] = deccomb;
            index++;            
            
            j=0;
            while((comb[j]+1)==comb[j+1]) {
                comb[j] = j;
                j++;
            }
            
            if(j<t) {
                comb[j]++;
            }
            
        }
        
        return combset;
      
    }
    
    /**
     * Generates all the (s+t)-bit strings with Hamming weight t, in binary
     * notation. The algorithm is described in Knuth, "The Art of Computer
     * Programming, pre-Fascicle 4A" (Algorithm L, p. 4).
     * 
     * @param s         number of 0s in the bitstrings
     * @param t         number of 1s in the bitstrings
     * @return          an array of boolean arrays, where each array is composed
     *                  of s false values and t true values
     */
    public static boolean[][] genBinCombsBin(int s, int t) {

        int size = binCoeff(s+t, t);
        boolean[][] combset = new boolean[size][];
        
        int index = 0; //index for the set combs.
        
        //Initialisation
        int[] comb = new int[t+2]; //the two additional cells are sentinels.
        for(int j=0; j<t; j++) {
            comb[j] = j;
        }
        comb[t] = s+t;
        comb[t+1] = 0;
        
        int j = 0;
        
        while(j<t) {
            
            boolean[] conf = new boolean[s+t];
            
            for(int k=0; k<t; k++) {
                conf[comb[k]] = true;
            }
            
            //Copy the combination in the final set.
            //int deccomb = bin2Dec(conf);
            combset[index] = conf;
            index++;            
            
            j=0;
            while((comb[j]+1)==comb[j+1]) {
                comb[j] = j;
                j++;
            }
            
            if(j<t) {
                comb[j]++;
            }
            
        }
        
        return combset;
      
    }
    
    /**
     * Creates a set of indices ordered by increasing Hamming weights. Given
     * the maximum weight as an input parameter, the method creates an int
     * matrix, where for all i between 0 (included) and maxweight (excluded) the
     * i-th row is an array holding the int indices whose binary representation
     * have Hamming weight i+1.
     * 
     * Notice that, in the order above, weight 0 is not considered (so indices[0]
     * will hold the indices of Hamming weight 1).
     * 
     * This method can be used in the computation of cryptographic properties
     * such as correlation immunity and propagation criteria, since they are
     * defined by subsets of binary vectors up to a certain Hamming weight.
     * 
     * 
     * @param maxweight     Maximum Hamming weight to be reached in the creation
     *                      of the indices
     * @return              An int matrix where each row i in [0..maxweight-1] is
     *                      an int array holding the integer representations of
     *                      the binary vectors of Hamming weight i+1
     * 
     */
    public static int[][] createIndices(int maxweight) {
        
        int[][] indices = new int[maxweight][];
        for(int i=0; i<maxweight; i++) {
            
            indices[i] = genBinCombs(maxweight-i-1, i+1);
            
        }
        
        return indices;
        
    }
    
    /**
     * Convert a boolean array in an int array of 0s (-> false) and 1s (-> true)
     * 
     * @param boolStr   A boolean array to be converted in int array
     * @return          An int array representing the conversion of boolStr in
     *                  0s (false) and 1s (true)
     */
    public static int[] bool2Int(boolean[] boolStr) {
        
        int[] intStr = new int[boolStr.length];
        
        for(int i=0; i<boolStr.length; i++) {
            
            if(boolStr[i]) {
                
                intStr[i] = 1;
                
            }
            
        }
        
        return intStr;
        
    }
    
    /**
     * Convert an int array composed of only 0s and 1s in a boolean array
     * (0 -> false, 1 -> true).
     * 
     * @param intStr    An int array composed only of 0s and 1s to be converted
     *                  in a boolean array
     * @return          A boolean array representing the conversion of intStr
     *                  (0 -> false, 1 -> true)
     */
    public static boolean[] int2Bool(int[] intStr) {
        
        boolean[] boolStr = new boolean[intStr.length];
        
        for(int i=0; i<intStr.length; i++) {
            
            if(intStr[i] == 1) {
                
                boolStr[i] = true;
                
            }
            
        }
        
        return boolStr;
        
    }
    
        /**
     * 
     * Returns a binary string in polar form (false -> 1, true -> -1)
     * 
     * @param   boolStr a boolean array representing the binary string
     * @return          an int array representing the polar form of the string
     */
    public static int[] bin2Pol(boolean[] boolStr) {

        int[] polStr = new int[boolStr.length];

        for(int i=0; i<boolStr.length; i++) {
            if(boolStr[i])
                polStr[i] = -1;
            else
                polStr[i] = 1;
        }

        return polStr;
    }
    
    public static int[] bin2PolInt(int[] boolStr) {

        int[] polStr = new int[boolStr.length];

        for(int i=0; i<boolStr.length; i++) {
            if(boolStr[i] == 1)
                polStr[i] = -1;
            else
                polStr[i] = 1;
        }

        return polStr;
    }
    
    /**
     * 
     * Returns a polar string in binary form (1 -> false, -1 -> true)
     * 
     * @param   polStr     a boolean array representing the binary string
     * @return             an int array representing the polar form of the string
     */
    public static boolean[] pol2Bin(int[] polStr) {

        boolean[] boolStr = new boolean[polStr.length];

        for(int i=0; i<polStr.length; i++) {
            
            if(polStr[i]==-1) {
            
                boolStr[i] = true;
                
            }
            
        }

        return boolStr;
        
    }
    
    public static int[] pol2BinInt(int[] polStr) {

        int[] boolStr = new int[polStr.length];

        for(int i=0; i<polStr.length; i++) {
            
            if(polStr[i]==-1) {
            
                boolStr[i] = 1;
                
            }
            
        }

        return boolStr;
        
    }
    
    /**
     * Converts a decimal number (BigInteger format) in a n-ary array of int.
     * 
     * @param   dNum    The BigInteger decimal number to convert
     * @param   length  The length of the array necessary to hold the n-ary
     *                  representation of dNum
     * @param   n       The radix for conversion
     * @return          An array of int holding the n-ary representation of dNum
     */
    public static int[] dec2Nary(BigInteger dNum, int length, int n) {
        
        int[] enNum = new int[length];
        BigInteger temp = dNum;
        BigInteger en = new BigInteger(Integer.toString(n));
        int i = 0;

        //Main loop: continue to divide temp by n until we reach zero, and
        //save the remainders in the n-ary array enNum
        while(temp.compareTo(BigInteger.ZERO) != 0) {

            BigInteger mod = temp.remainder(en);
            temp = temp.divide(en);
            //the array is filled in reverse order, since we are using MSBF
            enNum[enNum.length-i-1] = Integer.parseInt(mod.toString());

            i++;

        }

        return enNum;

    }
    
    /**
     * Converts a decimal number in int format in a n-ary array of int.
     * 
     * @param   dNum    The int decimal number to convert
     * @param   length  The length of the array necessary to hold the n-ary
     *                  representation of dNum
     * @param   n       The conversion radix
     * @return          An array of int holding the n-ary representation of dNum
     */
    public static int[] dec2Nary(int dNum, int length, int n) {
        
        int[] enNum = new int[length];
        int temp = dNum;
        int i = 0;

        //Main loop: continue to divide temp by n until we reach zero, and
        //save the remainders in the n-ary array enNum
        while(temp != 0) {

            int mod = temp % n;
            temp = temp / n;
            //the array is filled in reverse order, since we are using MSBF
            enNum[enNum.length-i-1] = mod;

            i++;

        }

        return enNum;

    }
    
    /**
     * Wrapper method to convert a BigInteger number in binary, represented as
     * a boolean array. The method first calls dec2Nary with radix 2 and then
     * transforms the resulting int array in a boolean array.
     * 
     * @param   dNum    The BigInteger decimal number to convert in binary
     * @param   length  The length of the array necessary to hold the binary
     *                  representation of dNum
     * @return          An array of boolean holding the binary representation of
     *                  dNum (0 -> false, 1 -> true)
     */
    public static boolean[] dec2Bin(BigInteger dNum, int length) {
        
        int[] binNum = dec2Nary(dNum, length, 2);
        boolean[] boolNum = int2Bool(binNum);
        
        return boolNum;
        
    }
    
    /**
     * Wrapper method to convert a int number in binary, represented as
     * a boolean array. The method first calls dec2Nary with radix 2 and then
     * transforms the resulting int array in a boolean array.
     * 
     * @param   dNum    The int number to convert in binary
     * @param   length  The length of the array necessary to hold the binary
     *                  representation of dNum
     * @return          An array of boolean holding the binary representation of
     *                  dNum (0 -> false, 1 -> true)
     */
    public static boolean[] dec2Bin(int dNum, int length) {
        
        int[] binNum = dec2Nary(dNum, length, 2);
        boolean[] boolNum = int2Bool(binNum);
        
        return boolNum;
        
    }
    
    /**
     * Wrapper method to convert a BigInteger number in hexadecimal, represented
     * as a String. The method first calls dec2Nary() with radix 16 and then
     * transforms the resulting int array in a String by calling nAry2Hex().
     * 
     * @param dNum      The BigInteger number to convert in hexadecimal
     * @param length    The length of the array necessary to hold the hexadecimal
     *                  representation of dNum
     * @return          An string holding the hexadecimal conversion of dNum
     */
    public static String dec2Hex(BigInteger dNum, int length) {
        
        int[] hexNum = dec2Nary(dNum, length, 16);
        String hexStr = nAry2Hex(hexNum);
        
        return hexStr;
        
    }
    
    /**
     * Wrapper method to convert an int number in hexadecimal, represented
     * as a String. The method first calls dec2Nary() with radix 16 and then
     * transforms the resulting int array in a String by calling nAry2Hex().
     * 
     * @param dNum      The int number to convert in hexadecimal
     * @param length    The length of the array necessary to hold the hexadecimal
     *                  representation of dNum
     * @return          An string holding the hexadecimal conversion of dNum
     */
    public static String dec2Hex(int dNum, int length) {
        
        int[] hexNum = dec2Nary(dNum, length, 16);
        String hexStr = nAry2Hex(hexNum);
        
        return hexStr;
        
    }
    
    /**
     * Converts a n-ary string in a decimal number (BigInteger version).
     * 
     * @param   enNum an array of int holding the n-ary representation of a number (MSBF order)
     * @param   n radix of the n-ary number
     * @return  a BigInteger representing the conversion of enNum as a decimal number
     */
    public static BigInteger nAry2DecBig(int[] enNum, int n) {
        
        BigInteger dNum = new BigInteger("0");
        BigInteger en = new BigInteger(Integer.toString(n));
        
        for(int i=0; i<enNum.length; i++) {

            BigInteger toAdd = en;
            toAdd = toAdd.pow(i);
            //The number is converted in reverse order, since it is represented in MSBF
            toAdd = toAdd.multiply(new BigInteger(Integer.toString(enNum[enNum.length-i-1])));
            dNum = dNum.add(toAdd);
             
        }

        return dNum;
        
    }
    
    /**
     * Converts a n-ary string in a decimal number (int version).
     * 
     * @param   enNum an array of int holding the n-ary representation of a number (MSBF order)
     * @param   n radix of the n-ary number
     * @return  an int representing the conversion of enNum as a decimal number
     */
    public static int nAry2DecInt(int[] enNum, int n) {
        
        int dNum = 0;
        
        for(int i=enNum.length-1; i>=0; i--) {
            
            //The number is converted in reverse order, since it is represented in MSBF
            dNum = dNum + enNum[enNum.length-i-1]* (int)Math.pow(n, i);
             
        }

        return dNum;
        
    }
    
    /**
     * Convert an hex number in its n-ary form (n=16), as an array of int.
     * 
     * @param hexnum    String representation of the hexadecimal number
     * @return          An array of int holding the 16-ary representation of hexnum.
     */
    public static int[] hex2nAry(String hexnum) {
        
        int[] nary = new int[hexnum.length()];
        String hexdigits = "0123456789ABCDEF";
        
        for(int i=0; i<hexnum.length(); i++) {
            
            char c = hexnum.charAt(i);
            nary[i] = hexdigits.indexOf(c);
            
        }
        
        return nary;
        
    }
    
    /**
     * Convert an array of int in base 16 in the corresponding hex number.
     * 
     * @param nary      An array of int holding the 16-ary representation of a hex number.
     * @return          A string corresponding to the hex number represented by nary
     */
    public static String nAry2Hex(int[] nary) {
        
        String hexnum = "";
        String hexdigits = "0123456789ABCDEF";
        
        for(int i=0; i<nary.length; i++) {
            
            hexnum += hexdigits.charAt(nary[i]);
            
        }
        
        return hexnum;
        
    }
    
    /**
     * Wrapper method to convert an hex number in String form to a BigInteger
     * decimal number.
     * 
     * @param hexnum    String representation of the hexadecimal number
     * @return          A BigInteger holding the decimal representation of hexnum
     */
    public static BigInteger hex2DecBig(String hexnum) {
        
        //Convert the string in n-ary int format, n=16
        int[] nary = hex2nAry(hexnum);
        
        //Convert the n-ary array in a BigInteger number
        BigInteger dNum = nAry2DecBig(nary, 16);
        
        return dNum;
        
    }
    
    /**
     * Wrapper method to convert an hex number in String form to an int
     * decimal number.
     * 
     * @param hexnum    String representation of the hexadecimal number
     * @return          An int holding the decimal representation of hexnum
     */
    public static int hex2DecInt(String hexnum) {
        
        //Convert the string in n-ary int format, n=16
        int[] nary = hex2nAry(hexnum);
        
        //Convert the n-ary array in an int number
        int dNum = nAry2DecInt(nary, 16);
        
        return dNum;
        
    }
    
    /**
     * Convert an hex string in a binary (boolean) array.
     * 
     * @param hexnum    String representation of the hexadecimal number
     * @return          An array of boolean holding the binary representation of hexnum.
     */
    public static boolean[] hex2Bin(String hexnum) {
        
        //The length of the boolean array is 4 times that of the hex string,
        //since each hex digit corresponds to 4 bits
        boolean[] boolNum = new boolean[hexnum.length()*4];
        int[] nary = hex2nAry(hexnum);
        
        //Convert each hex digit (in int form) in a group of 4 bits
        for(int i=0; i<nary.length; i++) {
            
            //Convert the single hex digit in binary form
            boolean[] bdigit = dec2Bin(nary[i], 4);
            
            //Copy the 4 bits in the corresponding portion of boolNum, which
            //starts at index i*4
            System.arraycopy(bdigit, 0, boolNum, i*4, bdigit.length);
            
        }
        
        
        return boolNum;
        
    }
    
    /**
     * Wrapper method to convert a binary number represented in a boolean array
     * in a decimal number (BigInteger format).
     * 
     * @param bNum      A boolean array holding the binary number to convert
     * @return          A BigInteger representing the binary number in decimal notation
     */
    public static BigInteger bin2DecBig(boolean[] bNum) {

        int[] nary = bool2Int(bNum);
        BigInteger dNum = nAry2DecBig(nary, 2);
        
        return dNum;
        
    }
    
    /**
     * Wrapper method to convert a binary number represented in a boolean array
     * in a decimal number (int format).
     * 
     * @param bNum      A boolean array holding the binary number to convert
     * @return          An int representing bNum in decimal notation
     */
    public static int bin2DecInt(boolean[] bNum) {

        int[] nary = bool2Int(bNum);
        int dNum = nAry2DecInt(nary, 2);
        
        return dNum;
        
    }
    
    /**
     * Wrapper method to convert a binary number represented in a boolean array
     * in a hexadecimal number (String format).
     * 
     * @param bNum      A boolean array holding the binary number to convert
     * @return          A string representing bNum in hexadecimal format
     */
    public static String bin2Hex(boolean[] bNum) {
        
        BigInteger dNum = bin2DecBig(bNum);
        int hexlen = (int)bNum.length/4;
        int[] hexnary = dec2Nary(dNum, hexlen, 16);
        String hexnum = nAry2Hex(hexnary);
        
        return hexnum;
        
    }
    
    /**
     * Converts a single boolean value in a 0 (false) or 1 (true).
     * 
     * @param   bval a boolean value.  
     * @return       1 if bval is true, 0 otherwise
     */
    public static int singleBool2Bin(boolean bval) {
        
        if(bval)
            return 1;
        else
            return 0;
        
    }

    /**
     * Converts a binary string represented as a boolean array in a
     * corresponding string of 0s and 1s.
     * 
     * @param   boolstr the binary string represented as a boolean array
     * @return          the binary string represented as string of 0s and 1s.
     */
    public static String bool2Bin(boolean[] boolstr) {
            
        String binstr = "";

        for(int i=0; i<boolstr.length; i++) {
                
            if(boolstr[i])
                binstr += "1";
            else
                binstr += "0";
            
        }

        return binstr;
            
    }
    
    public static boolean[] bin2Bool(String binstr) {
        
        boolean[] boolstr = new boolean[binstr.length()];
        
        for(int i=0; i<binstr.length(); i++) {
            
            if(binstr.charAt(i)=='1') {
                boolstr[i] = true;
            }
            
        }
        
        return boolstr;
        
    }
    
    public static void printArray(int[] v) {
        System.out.print("[");
        for (int i = 0; i < v.length; i++){
            System.out.print(v[i]);
            if (i != v.length - 1) {
                System.out.print(", ");
            }
        }
        
        System.out.print("]\n");
    }
    
    public static int balancing(int[] v){
        int numZeros = 0;
        int numOnes = 0;
        for (int i = 0; i < v.length; i++){
            if (v[i] == 1){
                numOnes++;
            }
            if (v[i] == 0){
                numZeros++;
            }
        }
        return Math.abs(numOnes - numZeros);
    }
    
    public static int resiliency(int[] spectrum){
        int max_resiliency_found_so_far = -1;
        int n_bits = (int)(Math.log(spectrum.length) / Math.log(2));
        int t = 0;
        if (spectrum[0] == 0){
            max_resiliency_found_so_far = 0;
            for (int iii = 0; iii < n_bits; iii++){
                t++;
                boolean stopHere = false;
                for (int kk = 0; kk < spectrum.length; kk++){
                    String binaryStr = Integer.toBinaryString(kk);
                    int hamming = 0;
                    for (int kkk=0; kkk < binaryStr.length(); kkk++){
                        if (binaryStr.charAt(kkk) == '1'){
                            hamming++;
                        }
                    }
                    if (hamming <= t){
                        if (spectrum[kk] != 0){
                            stopHere = true;
                            break;
                        }
                    }
                }
                if (stopHere)
                    return max_resiliency_found_so_far;
                else{
                    max_resiliency_found_so_far++;
                    if (iii == n_bits - 1){
                        return max_resiliency_found_so_far;
                    }
                }
                
            }
        }
        
        return max_resiliency_found_so_far;
        
    }
    
    
    
    public static void main(String[] args) {
    	
    	Random r = new Random(1);
    	int n_bits = 10;
    	int[][] a = generateKRandomTruthTables(r, n_bits, 10);
    	int i = 1;
    	printArray(a[i]);
    	int balan = balancing(a[i]);
    	int[] v = bin2PolInt(a[i]);
    	int spectralRadius = computeFWT(v, 0, v.length);
    	printArray(v);
    	int m = -50000000;
    	for (int j = 0; j < v.length; j++){
    	    if (Math.abs(v[j]) > m){
    	        m = Math.abs(v[j]);
    	    }
    	}
    	int res = resiliency(v);
    	int maxAutoCorrelationCoefficient = computeInvFWT(v, 0, v.length);
    	v = pol2BinInt(v);
    	if (!Arrays.equals(a[i], v)){
    	    throw new RuntimeException("Error walsh");
    	}
    	if (m != spectralRadius){
    	    throw new RuntimeException("Spectral radius wrong");
    	}
    	if (maxAutoCorrelationCoefficient != 1){
    	    throw new RuntimeException("Max autocorrelation coefficient wrong.");
    	}
    	boolean[] b = new boolean[v.length];
    	for (int j = 0; j<b.length; j++){
    	    if (v[j] == 1)
    	        b[j] = true;
    	}
    	int degree = computeFMT(b, 0, v.length);
    	int[] vv = new int[v.length];
    	for (int j = 0; j<b.length; j++){
    	    if (b[j])
    	        vv[j] = 1;
    	}
    	printArray(vv);
    	System.out.print("Degree: ");
    	System.out.print(degree);
    	System.out.print(" - Balancedness: ");
    	System.out.print(balan);
    	System.out.print(" - NonLinearity: ");
    	System.out.print((int)(Math.pow(2, n_bits - 1) - 0.5 * m));
    	System.out.print(" - Resiliency: ");
    	System.out.print(res);
    	System.out.print(" - SpectralRadius: ");
    	System.out.print(spectralRadius);
    	System.out.print(" - MaxAutoCorrelationCoefficient: ");
    	System.out.print(maxAutoCorrelationCoefficient);
    	System.out.println();
    	int[] x = {4, 4, 9, 4, 4, 4, 9, 4, 1, 0, 9, 1, 1, 0, 9, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 4, 1, 1, 1, 4, 1, 1};
    	maxAutoCorrelationCoefficient = computeInvFWT(x, 0, x.length);
    	printArray(x);
    	
    
    	
    }
    
// [-3 -1 -1  0 -1 -3  0 -1]
// [-1  0 -1  0  0  0  0  0]

// [-1  9  1 11  9  3 11  5]
// [ 6 -1 -1  0 -1 -4  0  0]

// [-4  4  0  0  8  8  8  8  4 -4  0  0 -8 -8 -4 -4  4  0  8 -4  0  4 -4  0 -4  0 -8  4  0 -4  4  0]
// [ 0  0  0  0  0  0  0  0  2  0  1 -1 -1  1  0  0  0  0  0  0  0  0  0  0 1 -1  0  0 -2 -2  0  0]

// [4 4 9 4 4 4 9 4 1 0 9 1 1 0 9 1 0 0 1 0 0 0 1 0 1 4 1 1 1 4 1 1]
// [ 2  1 -1 -1  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0  0  0  0  0  0 0  0  0  0  0  0  0  0]

// [ 4 -4  4  4 -4 -4  4  4 -4 -4 -4 -4 -4 -4 -4 -4]
// [-1 0 -1 0 0 0 0 0 2 0 -1 0 0 0 0 0]

// [ 4  4  4  4 -4 -4  4 -4  4  4  4  4  4 -4  4 -4]
// [1 1 0 0 2 -1 0 0 0 0 0 0 0 0 0 0]

// [-4 -4  4  4 -4 -4 -4 -4 -4 -4 -4 -4  4  4 -4 -4]
// [-2  0  0  0  0  0 -2  0  0  0 -2  0  2  0  0  0]

// [-4 -4 -4 -4  4  4 -4 -4  4  4 -4 -4  4  4 -4 -4]
// [-1  0  3  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0]

// [-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4]
// [-4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

// [-4  4 -4 -4  4  4 -4  4 -4  4 -4 -4  4  4 -4  4]
// [ 0 -2  2  0 -2  0  0 -2  0  0  0  0  0  0  0  0]

// [ 4  4  4  4 -4  4  4  4  4  4  4 -4  4  4  4  4]
// [ 3  0  0 -1  0  1  1  0  0 -1 -1  0  1  0  0  1]

// [ 4 -4  4  4 -4 -4  4  4  4 -4  4  4 -4 -4  4  4]
// [ 1  1 -3  1  1  1  1  1  0  0  0  0  0  0  0  0]

// [ 4 -4  4  4 -4  4 -4 -4  4 -4  4  4 -4  4 -4 -4  4 -4  4  4 -4  4  4 -4 4 -4  4  4  4  4 -4 -4]
// [ 0  0  0  0  1  1 -2  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

// [5 5 1 1 5 5 1 1 1 5 5 1 1 5 5 1]
// [ 3  0  1 -1  0  0  0  0  0  0  1  1  0  0  0  0]

// [ -999999998          -2          -2  -999999999           0          -2 -2          -2 -1000000000           0           0  -999999999 0           0           0           0]
// [-250000000          0          0 -249999999 -249999999          0 0 -249999999          0          0          0          0  0          0          0          0]

// [-8 -8 -8 -8  8  8  8  8  8  8  8  8  8  8  8  8 -8 -8 -8 -8 -8 -8 -8 -8 8 -8  8 -8  8  8  8  8 -8 -8 -8 -8  8  8  8  8  8  8  8  8  8  8  8  8 -8  8 -8  8 -8  8 -8  8  8  8  8  8  8  8  8  8]
// [ 2  0  0  0 -2  0  0  0 -4 -1  0  0 -1  0  0  0  1  0  0  0 -1  0  0  0 0  1  0  0 -2  0  0  0 -1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

// [-1  3  3 -1 -1  3  3 -1 -1  3  3 -1 -1  3  3 -1 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -3  1  1 -3 -1  3  3 -1 -1  3  3 -1 -1  3  3 -1 -1  3  3 -1]
// [ 0  0  0 -2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
  


}



