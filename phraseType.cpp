#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <time.h>
#include <cstdio>
#include <string.h>
#include <assert.h>

using namespace std;

// Paramter settings.
int N = 3; // |A| - # aspects

// Dirichlet priors.
double alpha = 50/N;
double beta = 0.01;
double beta_lrp = 0.01;
double beta_rrp = 0.01;
int numIter = 500;

// Flags to use certain features alone. (By default, all are used)
int useRP = 1;
int useWords = 1;
int useSP = 1;

// Global variable definitions.
int wordVocabSize;
int rpVocabSize;
int spVocabSize;

int P; // |P| - # phrases

// Standard vocab definitions
map<string,int> wordMap;
map<int,string> wordMapRev;
map<string,int> spMap;
map<int,string> spMapRev;
map<string,int> rpMap;
map<int,string> rpMapRev;

set<string> wordVocab;
set<string> spVocab;
set<string> rpVocab;

// Input data - list of words, list of sp, list of lrp, list of rrp.

vector<vector<int> > phrases_word; // list of words for each phrase.
vector<vector<int> > phrases_sp; // list of sig. phrases for each phrase.
std::vector<int> phrases_lrp; // lrp for each phrase.
std::vector<int> phrases_rrp; // rrp for each phrase.

// Variables and counters for gibbs sampler.
vector<int> assignments; // aspect assignment for each phrase - vector of size |P|.

// Aggregate counters.
int *nz_phrases; // #  phrases assigned to aspect z -  array of size |A|
int *nz_words_count; // # words assigned to aspect z -  array of size |A|
int *nz_sp_count; // # sig phrases assigned to aspect z -  array of size |A|
int *nz_lrp_count; // # lrp assigned to aspect z -  array of size |A|
int *nz_rrp_count; // # rrp assigned to aspect z -  array of size |A|

// Counters
int **nwz; // # times word w was assigned to aspect z - size |Vw| x |A|
int **nspz; // # times word w was assigned to aspect z - size |Vsp| x |A|
int **n_lrp_z; // # times specific lrp was assigned to aspect z - |Vrp| x |A|
int **n_rrp_z; // # times specific rrp was assigned to aspect z - |Vrp| x |A|

int **nzw; // number of times aspect z is assignment to word w.
int **nzsp; // number of times aspect z is assignment to sig phrase sp.
int **nz_lrp;
int **nz_rrp;

// Helper utilities for string split.
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

// Read input file to create vocabularies.
void readVocab(char* filename){
    int counter = 0;
    ifstream infile(filename);
    string line;
    while (getline(infile,line)){
          // Each line is a phrase with additional details.
          std::vector<string> phrase_parts = split(line,'$');
          assert(phrase_parts.size() == 5);
          string lrp = phrase_parts[0];
          std::vector<string> words = split(phrase_parts[1],',');
          std::vector<string> sig_phrases = split(phrase_parts[2], ',');
          string rrp = phrase_parts[3];
          string venue = phrase_parts[4];

          for(int w = 0; w < words.size(); w++)
              wordVocab.insert(words[w]);
          for (int sp = 0; sp < sig_phrases.size(); sp++)
              spVocab.insert(sig_phrases[sp]);

          rpVocab.insert(lrp);
          rpVocab.insert(rrp);
    }

    // Create word map and rev
    cout << rpVocab.size() << endl;
    set<string>::iterator it;
    for (it = wordVocab.begin(); it != wordVocab.end(); it++){
        wordMap.insert(pair<string,int>(*it, counter));
        wordMapRev.insert(pair<int,string>(counter, *it));
        counter++;
    }

    // Create sp map and rev
    counter = 0;
    for (it = spVocab.begin(); it != spVocab.end(); it++){
        spMap.insert(pair<string,int>(*it, counter));
        spMapRev.insert(pair<int,string>(counter, *it));
        counter++;
    }

    // Create rp map and rev
    counter = 0;
    for (it = rpVocab.begin(); it != rpVocab.end(); it++){
        rpMap.insert(pair<string,int>(*it, counter));
        rpMapRev.insert(pair<int,string>(counter, *it));
        counter++;
    }
}
// Read phrases from input file.
void readPhrases(char* filename){
    ifstream infile(filename);
    string line;

    while (getline(infile,line)){
        // Each line is a phrase.
        std::vector<string> phrase_parts = split(line,'$');
        string lrp = phrase_parts[0];
        std::vector<string> words = split(phrase_parts[1],',');
        std::vector<string> sig_phrases = split(phrase_parts[2], ',');
        string rrp = phrase_parts[3];
        string venue = phrase_parts[4];
        phrases_lrp.push_back(rpMap.find(lrp)->second);
        phrases_rrp.push_back(rpMap.find(rrp)->second);

        vector<int> words_int;
        for (int i = 0; i < words.size(); i++)
            words_int.push_back(wordMap.find(words[i])->second);
        phrases_word.push_back(words_int);

        vector<int> sp_int;
        for (int i = 0; i < sig_phrases.size(); i++)
            sp_int.push_back(spMap.find(sig_phrases[i])->second);
        phrases_sp.push_back(sp_int);
    }
    P = phrases_word.size();
}

// Defintion and Initialization of counters.
void initAssign(char* filename){
    // filename is the phrases file.
    readVocab(filename);  // read all vocabs.
    readPhrases(filename); // read phrases here.
    rpVocabSize = rpVocab.size();
    wordVocabSize = wordVocab.size();
    spVocabSize = spVocab.size();
    cout << "Vocab size " << wordVocabSize <<endl;
    cout << "# phrases " << P <<endl;
    cout << "RP vocab size " << rpVocabSize << endl;
    cout << "SP vocab size " << spVocabSize << endl;

    // Define counters.
    nwz = new int*[wordVocabSize];
    nspz = new int*[spVocabSize];
    n_lrp_z = new int*[rpVocabSize];
    n_rrp_z = new int*[rpVocabSize];

    nzw = new int*[N];
    nzsp = new int*[N];
    nz_lrp = new int*[N];
    nz_rrp = new int*[N];

    nz_words_count = new int[N];
    nz_sp_count = new int[N];
    nz_lrp_count = new int[N];
    nz_rrp_count = new int[N];
    nz_phrases = new int[N];

    fill_n(nz_words_count, N, 0.0);
    fill_n(nz_lrp_count, N, 0.0);
    fill_n(nz_rrp_count, N, 0.0);
    fill_n(nz_phrases, N, 0.0);


    for (int i = 0; i < wordVocabSize; i++){
        nwz[i] = new int[N];
        fill_n(nwz[i], N, 0.0);
    }

    for (int i = 0; i < spVocabSize; i++){
        nspz[i] = new int[N];
        fill_n(nspz[i], N, 0.0);
    }

    for (int i = 0; i < rpVocabSize; ++i)
    {
        n_lrp_z[i] = new int[N];
        fill_n(n_lrp_z[i], N, 0.0);
        n_rrp_z[i] = new int[N];
        fill_n(n_rrp_z[i], N, 0.0);
    }
    for (int i = 0; i < N; i++){
        nzw[i] = new int[wordVocabSize];
        fill_n(nzw[i], wordVocabSize, 0.0);
        nzsp[i] = new int[spVocabSize];
        fill_n(nzsp[i], spVocabSize, 0.0);
        nz_lrp[i] = new int[rpVocabSize];
        fill_n(nz_lrp[i], rpVocabSize, 0.0);
        nz_rrp[i] = new int[rpVocabSize];
        fill_n(nz_rrp[i], rpVocabSize, 0.0);
    }

    // Do intial assignments randomly.
    for (int u = 0; u < phrases_word.size(); ++u)
    {
        vector<int> phrase_word = phrases_word[u];
        int t;
        t = rand()%N;
        assignments.push_back(t);

        nz_phrases[t] += 1;
        nz_words_count[t] += phrases_word[u].size();
        nz_sp_count[t] += phrases_sp[u].size();

        for (int w = 0; w < phrases_word[u].size(); w++){
            nwz[phrases_word[u][w]][t] += 1;
            nzw[t][phrases_word[u][w]] += 1;
        }

        for (int sp = 0; sp < phrases_sp[u].size(); sp++){
            nspz[phrases_sp[u][sp]][t] += 1;
            nzsp[t][phrases_sp[u][sp]] += 1;
        }

        if (phrases_lrp[u] != rpMap["empty"]){
            n_lrp_z[phrases_lrp[u]][t] += 1;
            nz_lrp[t][phrases_lrp[u]] += 1;
            nz_lrp_count[t] += 1;
        }
        if (phrases_rrp[u] != rpMap["empty"]){
            n_rrp_z[phrases_rrp[u]][t] += 1;
            nz_rrp[t][phrases_rrp[u]] += 1;
            nz_rrp_count[t] += 1;
        }
    }
}


void gibbsIteration(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> probs;

    for (int t = 0; t < N; ++t)
        probs.push_back(0.0);

    for (int u = 0; u < P; ++u)
    {
        // Iterate through each phrase.
        vector<int> words = phrases_word[u];
        vector<int> sig_phrases = phrases_sp[u];
        // Subtract current assignments before new assignments.
        int current_aspect = assignments[u];
        nz_phrases[current_aspect] -= 1;
        nz_words_count[current_aspect] -= words.size();
        nz_sp_count[current_aspect] -= sig_phrases.size();
        // lrp
        if (phrases_lrp[u] != rpMap["empty"]){
            nz_lrp[current_aspect][phrases_lrp[u]] -= 1;
            n_lrp_z[phrases_lrp[u]][current_aspect] -= 1;
            nz_lrp_count[current_aspect] -= 1;
        }
        // rrp
        if (phrases_rrp[u] != rpMap["empty"]){
            nz_rrp[current_aspect][phrases_rrp[u]] -= 1;
            n_rrp_z[phrases_rrp[u]][current_aspect] -= 1;
            nz_rrp_count[current_aspect] -= 1;
        }
        // words
        for (int w_iter = 0; w_iter < words.size(); ++w_iter)
        {
            int w = words[w_iter];
            nwz[w][current_aspect] -= 1;
            nzw[current_aspect][w] -= 1;
        }
        // sig phrases
        for (int sp_iter = 0; sp_iter < sig_phrases.size(); ++sp_iter)
        {
            int sp = sig_phrases[sp_iter];
            nspz[sp][current_aspect] -= 1;
            nzsp[current_aspect][sp] -= 1;
        }
        // Now sample aspect for each phrase.
        for (int t = 0; t < N; ++t)
        {
            probs[t] = ((nz_phrases[t]*1.0) + alpha)/(phrases_word.size() + N*alpha*1.0);
            for (int w_iter = 0; w_iter < words.size(); ++w_iter)
            {
                int w = words[w_iter];
                if(useWords == 1)
                    probs[t] *= (nwz[w][t]*1.0 + beta)/(nz_words_count[t] + (wordVocabSize*beta));
            }

            for (int sp_iter = 0; sp_iter < sig_phrases.size(); ++sp_iter)
            {
                int sp = sig_phrases[sp_iter];
                if(useSP == 1)
                    probs[t] *= (nspz[sp][t]*1.0 + beta)/(nz_sp_count[t] + (spVocabSize*beta));
            }

            if(useRP == 1)
            {
            	if(strcmp(rpMapRev[phrases_lrp[u]].c_str(), "empty") != 0 )
                	probs[t] *= (n_lrp_z[phrases_lrp[u]][t]*1.0+ beta_lrp)/(nz_lrp_count[t] + rpVocabSize* beta_lrp);
                if(strcmp(rpMapRev[phrases_rrp[u]].c_str(), "empty") != 0 )
                	probs[t] *= (n_rrp_z[phrases_rrp[u]][t]*1.0+ beta_rrp)/(nz_rrp_count[t] + rpVocabSize* beta_rrp);
            }
        }
        // Choose new aspect.
        std::discrete_distribution<> d(probs.begin(), probs.end());
        int new_aspect = d(gen);
        // Update counters based on chosen aspect.
        // words
        for (int w_iter = 0; w_iter < words.size(); ++w_iter)
        {
            int w = words[w_iter];
            nwz[w][new_aspect] += 1;
            nzw[new_aspect][w] += 1;
        }
        // sp
        for (int sp_iter = 0; sp_iter < sig_phrases.size(); ++sp_iter)
        {
            int sp = sig_phrases[sp_iter];
            nspz[sp][new_aspect] += 1;
            nzsp[new_aspect][sp] += 1;
        }
        // lrp
        if (phrases_lrp[u] != rpMap["empty"]){
            nz_lrp[new_aspect][phrases_lrp[u]] += 1;
            n_lrp_z[phrases_lrp[u]][new_aspect] += 1;
            nz_lrp_count[new_aspect] += 1;
        }
        // rrp
        if (phrases_rrp[u] != rpMap["empty"]){
            nz_rrp[new_aspect][phrases_rrp[u]] += 1;
            n_rrp_z[phrases_rrp[u]][new_aspect] += 1;
            nz_rrp_count[new_aspect] += 1;
        }

        nz_phrases[new_aspect] += 1;
        nz_words_count[new_aspect] += words.size();
        nz_sp_count[new_aspect] += sig_phrases.size();
        assignments[u] = new_aspect;
    }
}

void createOutputFiles(double** P_w_given_z, double** P_sp_given_z, double** P_lrp_given_z, double** P_rrp_given_z, double* Pz)
{
    ofstream F0;
    char path[40];
    strcpy(path, "PhraseType/");

    char f0_path[100];
    strcpy(f0_path, path);
    strcat(f0_path,"wordMap.txt");

    F0.open (f0_path);
    for (int i = 0; i < wordVocabSize; ++i)
        F0<<i<<" "<<wordMapRev.find(i)->second<<endl;
    F0.close();

    char f1_path[100];
    strcpy(f1_path, path);
    strcat(f1_path,"aspect_priors.txt");

    ofstream F1;
    F1.open (f1_path);
    for (int t = 0; t < N; ++t)
        F1<<Pz[t]<<endl;

    F1.close();
    char f2_path[100];
    strcpy(f2_path, path);
    strcat(f2_path,"word_aspect_probs.txt");

    ofstream F2;
    F2.open (f2_path);

    for (int t = 0; t < N; ++t)
    {
        F2 << P_w_given_z[t][0];
        for (int i = 1; i < wordVocabSize; ++i)
            F2<<" "<<P_w_given_z[t][i];
        F2 << endl;
    }
    F2.close();

    char f3_path[100];
    strcpy(f3_path, path);
    strcat(f3_path,"lrp_aspect_probs.txt");
    ofstream F3;
    F3.open (f3_path);

    for (int t = 0; t < N; ++t)
    {
        F3 << P_lrp_given_z[t][0];
        for (int i = 1; i < rpVocabSize; ++i)
            F3<<" "<<P_lrp_given_z[t][i];
        F3 << endl;
    }
    F3.close();

    char f4_path[100];
    strcpy(f4_path, path);
    strcat(f4_path,"rrp_aspect_probs.txt");
    ofstream F4;
    F4.open (f4_path);

    for (int t = 0; t < N; ++t)
    {
        F4 << P_rrp_given_z[t][0];
        for (int i = 1; i < rpVocabSize; ++i)
            F4<<" "<<P_rrp_given_z[t][i];
        F4 << endl;
    }
    F4.close();

    char f5_path[100];
    strcpy(f5_path, path);
    strcat(f5_path,"rpMap.txt");
    ofstream F5;
    F5.open (f5_path);
    for (int i = 0; i < rpVocabSize; ++i)
        F5<<i<<" "<<rpMapRev.find(i)->second<<endl;
    F5.close();

    char f6_path[100];
    strcpy(f6_path, path);
    strcat(f6_path,"spMap.txt");
    ofstream F6;
    F6.open (f6_path);
    for (int i = 0; i < spVocabSize; ++i)
      F6<<i<<" "<<spMapRev.find(i)->second<<endl;

    F6.close();
    char f7_path[100];
    strcpy(f7_path, path);
    strcat(f7_path,"sp_aspect_probs.txt");
    ofstream F7;
    F7.open (f7_path);

    for (int t = 0; t < N; ++t)
    {
        F7 << P_sp_given_z[t][0];
        for (int i = 1; i < spVocabSize; ++i)
            F7<<" "<<P_sp_given_z[t][i];
        F7 << endl;
    }
    F7.close();
}

void outputResult()
{
    double* Pz = new double[N]; // aspect dist. - P(c) - |A|
    double** P_lrp_given_z = new double*[N]; // P(lrp|z) - |A| x |Vrp|
    double** P_rrp_given_z = new double*[N]; // P(rrp|z) - |A| x |Vrp|

    for (int t = 0; t < N; ++t)
    {
        P_lrp_given_z[t] = new double[rpVocabSize];
        P_rrp_given_z[t] = new double[rpVocabSize];
        for (int i = 0; i < rpVocabSize; ++i)
        {
              P_lrp_given_z[t][i] = (nz_lrp[t][i] + beta_lrp)/ (nz_lrp_count[t] + rpVocabSize*beta_lrp);
              P_rrp_given_z[t][i] = (nz_rrp[t][i] + beta_rrp)/ (nz_rrp_count[t] + rpVocabSize*beta_rrp);
        }
    }

    double** P_w_given_z = new double*[N]; // P(w|z) - |A| x |Vw|
    double** P_sp_given_z = new double*[N]; // P(sp|z) - |A| x |Vsp|

    for (int t = 0; t < N; ++t)
    {
        P_w_given_z[t] = new double[wordVocabSize];
        P_sp_given_z[t] = new double[spVocabSize];

        for (int i = 0; i < wordVocabSize; ++i)
            P_w_given_z[t][i] = (1.0*(nzw[t][i]+ beta))/(nz_words_count[t] + wordVocabSize*beta);

        for (int i = 0; i < spVocabSize; ++i)
            P_sp_given_z[t][i] = (1.0*(nzsp[t][i]+ beta))/(nz_sp_count[t] + spVocabSize*beta);

        Pz[t] = (nz_phrases[t]+alpha)/(phrases_word.size() + N*alpha);
    }
    createOutputFiles(P_w_given_z,P_sp_given_z,P_lrp_given_z,P_rrp_given_z,Pz);
}


int main(int argc, char const *argv[])
{
    if(argc < 2)
      {
        cout << "Incorrect number of arguments" << endl;
        return 0;
      }
    srand(time(NULL));
    char phraseFile[50]; // Arg 1 is phrase file.
    strcpy(phraseFile, argv[1]);
    initAssign(phraseFile);
    clock_t t1,t2;
    t1=clock();
    for (int i = 0; i < numIter; ++i)
    {
        cout << i <<endl;
        gibbsIteration();
    }
    cout << "Gibbs done" <<endl;

    t2=clock();
    float diff ((float)t2-(float)t1);
    float seconds = diff / CLOCKS_PER_SEC;
    seconds /= numIter;
    cout<<"Run time per iter = "<<seconds<<endl;
    outputResult();
    return 0;
}
