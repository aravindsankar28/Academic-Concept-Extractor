#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <time.h>
#include <assert.h>
using namespace std;

int numIter = 200;

// Flags to use certain features alone. (By default, all are used)
int useRP = 1;
int useWords = 1;
int useSP = 1;
int useVenue = 1;

int C = 3; // |A| - # aspects
int D = 7; // |D| - # domains

// Dirichlet priors.
double alpha_D = 50.0/D;
double alpha_C = 50.0/C;

double beta_lrp = 0.01;
double beta_rrp = 0.01;
double beta_v = 0.01;
double beta_w = 0.01;

int wordVocabSize;
int rpVocabSize;
int venueVocabSize;
int spVocabSize;

int P;  // |P| - # phrases

// Standard vocab definitions
map<string,int> wordMap;
map<int,string> wordMapRev;
map<string,int> spMap;
map<int,string> spMapRev;
map<string,int> rpMap;
map<int,string> rpMapRev;
map<string,int> venueMap;
map<int,string> venueMapRev;

set<string> rpVocab;
set<string> wordVocab;
set<string> venueVocab;
set<string> spVocab;

// Input data - list of phrases/words, list of lrp, list of rrp, list of venues.

vector<vector<int> > phrases_word; // list of words for each phrase.
vector<vector<int> > phrases_sp; // list of sig. phrases for each phrase.
std::vector<int> phrases_lrp; // lrp for each phrase.
std::vector<int> phrases_rrp; // rrp for each phrase.
std::vector<int> phrases_venues; // venue for each phrase.


std::vector<pair<int,int> > assignments; // domain, aspect assignment for each phrase - vector of size |P| x 2.

// Variables and counters for gibbs sampler.

// Aggregate counters.
int *nd_phrases; // # phrases assigned to domain d - array of size |D|
int *nd_venues_count; // # venues assigned to domain d - array of size |D|
int *nc_phrases; // # phrases assigned to aspect c - array of size |A|
int *nc_lrp_count; // # lrp assigned to aspect c - array of size |A|
int *nc_rrp_count; // # rrp assigned to aspect c - array of size |A|
int **ndc_w_count; // # words assigned to aspect c and domain d - size |D| x |A|
int **ndc_sp_count; // # sig phrases assigned to aspect c and domain d - size |D| x |A|

// Counters
int **n_v_d; // # times specific venue v was assigned to domain d - size |Vv| x |D|
int **n_lrp_c; // # times specific lrp was assigned to aspect c - |Vrp| x |A|
int **n_rrp_c; // # times specific rrp was assigned to aspect c - |Vrp| x |A|
int ***n_w_dc; // # times word w was assigned to aspect c and domain d - |Vw| x |D| x |A|
int ***n_sp_dc; // # times sig phrase sp was assigned to aspect c and domain d - |Vsp| x |D| x |A|

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
          // assert(phrase_parts.size() == 5);
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
          venueVocab.insert(venue);
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

    // Create venue map and rev
    counter = 0;
    for (it = venueVocab.begin(); it != venueVocab.end(); it++){
        venueMap.insert(pair<string,int>(*it, counter));
        venueMapRev.insert(pair<int,string>(counter, *it));
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
        phrases_venues.push_back(venueMap.find(venue)->second);
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
    readPhrases(filename); // read phrases and stopwords here.
    rpVocabSize = rpVocab.size();
    wordVocabSize = wordVocab.size();
    venueVocabSize = venueVocab.size();
    spVocabSize = spVocab.size();
    cout << "Word Vocab size " << wordVocabSize <<endl;
    cout << "SP Vocab size " << spVocabSize <<endl;
    cout << "# phrases " << P <<endl;
    cout << "RP vocab size " << rpVocabSize << endl;
    cout << "# venues " << venueVocabSize << endl;

    // Define counters.
    nd_phrases = new int[D];
    nd_venues_count = new int[D];
    nc_phrases = new int[C];
    nc_lrp_count = new int[C];
    nc_rrp_count = new int[C];

    fill_n(nd_phrases, D, 0.0);
    fill_n(nd_venues_count, D, 0.0);
    fill_n(nc_phrases, C, 0.0);
    fill_n(nc_lrp_count, C, 0.0);
    fill_n(nc_rrp_count, C, 0.0);

    n_v_d  = new int*[venueVocabSize];
    n_lrp_c = new int*[rpVocabSize];
    n_rrp_c = new int*[rpVocabSize];
    n_w_dc = new int**[wordVocabSize];
    n_sp_dc = new int**[spVocabSize];

    ndc_w_count = new int*[D];
    ndc_sp_count = new int*[D];

    for (int i = 0; i < rpVocabSize; ++i)
    {
        n_lrp_c[i] = new int[C];
        fill_n(n_lrp_c[i], C, 0.0);
        n_rrp_c[i] = new int[C];
        fill_n(n_rrp_c[i], C, 0.0);
    }

    for (int i = 0 ; i < venueVocabSize; ++i)
    {
        n_v_d[i] = new int[D];
        fill_n(n_v_d[i], D, 0.0);
    }

    for (int i = 0; i < wordVocabSize; i++){
        n_w_dc[i] = new int*[D];
        for(int j = 0; j < D; j ++)
        {
            n_w_dc[i][j] = new int[C];
            fill_n(n_w_dc[i][j], C, 0.0);
        }
    }

    for (int i = 0; i < spVocabSize; i++){
        n_sp_dc[i] = new int*[D];
        for(int j = 0; j < D; j ++)
        {
            n_sp_dc[i][j] = new int[C];
            fill_n(n_sp_dc[i][j], C, 0.0);
        }
    }

    for(int i = 0; i < D; i++)
    {
        ndc_w_count[i] = new int[C];
        fill_n(ndc_w_count[i], C, 0.0);
        ndc_sp_count[i] = new int[C];
        fill_n(ndc_sp_count[i], C, 0.0);
    }

    // Do intial assignments randomly.
    for (int u = 0; u < phrases_word.size(); ++u)
    {
        int d, c;
        d = rand()%D;
        c = rand()%C;
        assignments.push_back(pair<int,int>(d, c));
        nc_phrases[c] += 1;
        nd_phrases[d] += 1;
        ndc_w_count[d][c] += phrases_word[u].size();
        ndc_sp_count[d][c] += phrases_sp[u].size();

        for (int w = 0; w < phrases_word[u].size(); w++){
            int word = phrases_word[u][w];
            n_w_dc[word][d][c] += 1;
        }

        for (int sp = 0; sp < phrases_sp[u].size(); sp++){
            int sig_phrase = phrases_sp[u][sp];
            n_sp_dc[sig_phrase][d][c] += 1;
        }

        if (phrases_lrp[u] != rpMap["empty"]){
            int lrp_phrase = phrases_lrp[u];
            n_lrp_c[lrp_phrase][c] += 1;
            nc_lrp_count[c] += 1;
        }
        if (phrases_rrp[u] != rpMap["empty"]){
            int rrp_phrase = phrases_rrp[u];
            n_rrp_c[rrp_phrase][c] += 1;
            nc_rrp_count[c] += 1;
        }
        n_v_d[phrases_venues[u]][d] += 1;
        nd_venues_count[d] +=1;
    }
}

void gibbsIteration(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> probs;

    for (int d = 0; d < D; ++d)
        for(int c = 0; c < C; ++c)
            probs.push_back(0.0);

    for (int p = 0; p < P; ++p)
    {
        // Iterate through each phrase.
        vector<int> phrase_words = phrases_word[p];
        vector<int> phrase_sp = phrases_sp[p];
        // Subtract current assignments before new assignments.
        int current_domain = assignments[p].first;
        int current_aspect = assignments[p].second;
        nc_phrases[current_aspect] -= 1;
        nd_phrases[current_domain] -= 1;
        ndc_w_count[current_domain][current_aspect] -= phrase_words.size();
        ndc_sp_count[current_domain][current_aspect] -= phrase_sp.size();
        // lrp
        if (phrases_lrp[p] != rpMap["empty"]){
            n_lrp_c[phrases_lrp[p]][current_aspect] -=1;
            nc_lrp_count[current_aspect] -=1;
        }
        // rrp
        if (phrases_rrp[p] != rpMap["empty"]){
            n_rrp_c[phrases_rrp[p]][current_aspect] -=1;
            nc_rrp_count[current_aspect] -=1;
        }
        // venue
        nd_venues_count[current_domain] -=1;
        n_v_d[phrases_venues[p]][current_domain] -=1;
        // words
        for (int w_iter = 0; w_iter < phrase_words.size(); ++w_iter)
        {
            int w = phrase_words[w_iter];
            n_w_dc[w][current_domain][current_aspect] -= 1;
        }
        // sig phrases
        for (int sp_iter = 0; sp_iter < phrase_sp.size(); ++sp_iter)
        {
            int sp = phrase_sp[sp_iter];
            n_sp_dc[sp][current_domain][current_aspect] -= 1;
        }
        // Now sample d,c pair for each phrase.
        for (int d = 0; d < D; ++d)
        {
            for (int c = 0; c < C; ++c)
            {
                int idx = d*C +c;
                probs[idx] = ((nd_phrases[d]*1.0) + alpha_D)/(phrases_word.size()-1 + D*alpha_D*1.0);
                probs[idx] *= ((nc_phrases[c]*1.0) + alpha_C)/(phrases_word.size()-1 + C*alpha_C*1.0);
                if(useRP == 1)
                {
                    if(strcmp(rpMapRev[phrases_lrp[p]].c_str(), "empty") != 0 )
                        probs[idx] *= (n_lrp_c[phrases_lrp[p]][c]*1.0+ beta_lrp)/(nc_lrp_count[c] + rpVocabSize*beta_lrp);

                    if(strcmp(rpMapRev[phrases_rrp[p]].c_str(), "empty") != 0 )
                        probs[idx] *= (n_rrp_c[phrases_rrp[p]][c]*1.0+ beta_rrp)/(nc_rrp_count[c] + rpVocabSize*beta_rrp);
                }
              if(useVenue == 1)
                probs[idx] *= (n_v_d[phrases_venues[p]][d]*1.0+ beta_v)/(nd_venues_count[d] + venueVocabSize*beta_v);

                for (int w_iter = 0; w_iter < phrase_words.size(); ++w_iter)
                {
                    int w = phrase_words[w_iter];
                    if(useWords == 1)
                        probs[idx] *= (n_w_dc[w][d][c]*1.0 + beta_w)/(ndc_w_count[d][c] + (wordVocabSize*beta_w));
                }

                for (int sp_iter = 0; sp_iter < phrase_sp.size(); ++sp_iter)
                {
                    int sp = phrase_sp[sp_iter];
                    if(useSP == 1)
                        probs[idx] *= (n_sp_dc[sp][d][c]*1.0 + beta_w)/(ndc_sp_count[d][c] + (spVocabSize*beta_w));
                }
            }
        }

        // Choose new aspect,domain.
        std::discrete_distribution<> d(probs.begin(), probs.end());
        // Now, sample from this distribution.
        int chosen_idx = d(gen);
        int new_domain = chosen_idx/C;
        int new_aspect = chosen_idx % C;

        // Update counters based on chosen d,c pair.
        nc_phrases[new_aspect] += 1;
        nd_phrases[new_domain] += 1;
        ndc_w_count[new_domain][new_aspect] += phrase_words.size();
        ndc_sp_count[new_domain][new_aspect] += phrase_sp.size();

        // lrp
        if (phrases_lrp[p] != rpMap["empty"]){
            n_lrp_c[phrases_lrp[p]][new_aspect] +=1;
            nc_lrp_count[new_aspect] +=1;
        }

        // rrp
        if (phrases_rrp[p] != rpMap["empty"]){
            n_rrp_c[phrases_rrp[p]][new_aspect] +=1;
            nc_rrp_count[new_aspect] +=1;
        }

        // venue
        nd_venues_count[new_domain] += 1;
        n_v_d[phrases_venues[p]][new_domain] +=1;

        // words
        for (int w_iter = 0; w_iter < phrase_words.size(); ++w_iter)
        {
            int w = phrase_words[w_iter];
            n_w_dc[w][new_domain][new_aspect] += 1;
        }
        // sp
        for (int sp_iter = 0; sp_iter < phrase_sp.size(); ++sp_iter)
        {
            int sp = phrase_sp[sp_iter];
            n_sp_dc[sp][new_domain][new_aspect] += 1;
        }
        assignments[p] =  pair<int,int>(new_domain, new_aspect);
    }
}


void createOutputFiles(double*Pd, double*Pc, double** P_lrp_given_c, double** P_rrp_given_c, double** P_v_given_d, double***P_w_given_dc,double***P_sp_given_dc)
{
    ofstream F0;
    char path[40];
    strcpy(path, "DomainPhraseType/");

    char f0_path[100];
    strcpy(f0_path, path);
    strcat(f0_path,"wordMap.txt");
    F0.open (f0_path);

    for (int i = 0; i < wordVocabSize; ++i)
        F0<<i<<" "<<wordMapRev.find(i)->second<<endl;

    F0.close();

    char f1_path[100];
    strcpy(f1_path, path);
    strcat(f1_path,"rpMap.txt");
    ofstream F1;
    F1.open (f1_path);
    for (int i = 0; i < rpVocabSize; ++i)
        F1<<i<<" "<<rpMapRev.find(i)->second<<endl;
    F1.close();

    char f2_path[100];
    strcpy(f2_path, path);
    strcat(f2_path,"venueMap.txt");
    ofstream F2;
    F2.open (f2_path);
    for (int i = 0; i < venueVocabSize; ++i)
        F2<<i<<" "<<venueMapRev.find(i)->second<<endl;
    F2.close();

    char f3_path[100];
    strcpy(f3_path, path);
    strcat(f3_path,"aspect_priors.txt");

    ofstream F3;
    F3.open (f3_path);
    for (int c = 0; c < C; ++c)
        F3<<Pc[c]<<endl;
    F3.close();

    char f4_path[100];
    strcpy(f4_path, path);
    strcat(f4_path,"domain_priors.txt");
    ofstream F4;
    F4.open (f4_path);
    for (int d = 0; d < D; ++d)
        F4<<Pd[d]<<endl;
    F4.close();

    char f5_path[100];
    strcpy(f5_path, path);
    strcat(f5_path,"lrp_aspect_probs.txt");
    ofstream F5;
    F5.open (f5_path);

    for (int c = 0; c < C; ++c)
    {
        F5 << P_lrp_given_c[c][0];
        for (int i = 1; i < rpVocabSize; ++i)
            F5<<" "<<P_lrp_given_c[c][i];
        F5 << endl;
    }
    F5.close();

    char f6_path[100];
    strcpy(f6_path, path);
    strcat(f6_path,"rrp_aspect_probs.txt");
    ofstream F6;
    F6.open (f6_path);

    for (int c = 0; c < C; ++c)
    {
        F6 << P_rrp_given_c[c][0];
        for (int i = 1; i < rpVocabSize; ++i)
            F6<<" "<<P_rrp_given_c[c][i];
        F6 << endl;
    }
    F6.close();

    char f7_path[100];
    strcpy(f7_path, path);
    strcat(f7_path,"venue_domain_probs.txt");
    ofstream F7;
    F7.open (f7_path);

    for (int d = 0; d < D; ++d)
    {
        F7 << P_v_given_d[d][0];
        for (int i = 1; i < venueVocabSize; ++i)
            F7<<" "<<P_v_given_d[d][i];
        F7 << endl;
    }
    F7.close();

    char f8_path[100];
    strcpy(f8_path, path);
    strcat(f8_path,"word_dc_probs.txt");

    ofstream F8;
    F8.open (f8_path);

    for (int d = 0; d < D; ++d)
        for (int c = 0; c < C; ++c)
        {
            F8 << d <<" "<<c <<" "<<P_w_given_dc[d][c][0];
            for (int i = 1; i < wordVocabSize; ++i)
                F8<<" "<<P_w_given_dc[d][c][i];
            F8 << endl;
        }

    F8.close();

    char f9_path[100];
    strcpy(f9_path, path);
    strcat(f9_path,"sp_dc_probs.txt");
    ofstream F9;
    F9.open (f9_path);

    for (int d = 0; d < D; ++d)
        for (int c = 0; c < C; ++c)
        {
            F9 << d <<" "<<c <<" "<<P_sp_given_dc[d][c][0];
            for (int i = 1; i < spVocabSize; ++i)
                F9<<" "<<P_sp_given_dc[d][c][i];
            F9 << endl;
        }

    F9.close();

    char f10_path[100];
    strcpy(f10_path, path);
    strcat(f10_path,"spMap.txt");
    ofstream F10;
    F10.open (f10_path);

    for (int i = 0; i < spVocabSize; ++i)
        F10<<i<<" "<<spMapRev.find(i)->second<<endl;

    F10.close();
}

void outputResult()
{
    double* Pd = new double[D]; // domain dist. - P(d) - |D|
    double* Pc = new double[C]; // aspect dist. - P(c) - |A|
    for (int d = 0; d < D; ++d)
        Pd[d] = (nd_phrases[d]+alpha_D)/(phrases_word.size()+ D*alpha_D);

    for (int c = 0; c < C; ++c)
        Pc[c] = (nc_phrases[c]+alpha_C)/(phrases_word.size()+ C*alpha_C);

    double** P_lrp_given_c = new double*[C]; // P(lrp|c) - |A| x |Vrp|
    double** P_rrp_given_c = new double*[C]; // P(rrp|c) - |A| x |Vrp|

    for(int c = 0; c < C; c++)
    {
        P_lrp_given_c[c] = new double[rpVocabSize];
        P_rrp_given_c[c] = new double[rpVocabSize];

        for(int i = 0; i < rpVocabSize; i++)
        {
            P_lrp_given_c[c][i] = (1.0*(n_lrp_c[i][c]+ beta_lrp))/(nc_lrp_count[c] + rpVocabSize *beta_lrp);
            P_rrp_given_c[c][i] = (1.0*(n_rrp_c[i][c]+ beta_rrp))/(nc_rrp_count[c] + rpVocabSize *beta_rrp);
        }
    }

    double** P_v_given_d = new double*[D]; // P(v|d) - |D| x |Vv|
    for(int d =0; d < D; d++)
    {
        P_v_given_d[d] = new double[venueVocabSize];
        for (int i = 0; i < venueVocabSize; ++i)
            P_v_given_d[d][i] = (1.0*(n_v_d[i][d]+ beta_v))/(nd_venues_count[d] + venueVocabSize * beta_v);
    }

    // Compute P(w | d,c) - |D| x |A| x |Vw|
    double ***P_w_given_dc = new double**[D];
    for (int d = 0; d < D; ++d)
    {
        P_w_given_dc[d] = new double*[C];
        for (int c = 0; c < C; ++c)
        {
            P_w_given_dc[d][c] = new double[wordVocabSize];
            for(int i = 0; i < wordVocabSize; i ++)
                P_w_given_dc[d][c][i] = (1.0*(n_w_dc[i][d][c] + beta_w))/(ndc_w_count[d][c] + wordVocabSize*beta_w);
        }
    }
    // Compute P(sp| d,c) - |D| x |A| x |Vsp|
    double ***P_sp_given_dc = new double**[D];
    for (int d = 0; d < D; ++d)
    {
        P_sp_given_dc[d] = new double*[C];
        for (int c = 0; c < C; ++c)
        {
            P_sp_given_dc[d][c] = new double[spVocabSize];
            for(int i = 0; i < spVocabSize; i ++)
                P_sp_given_dc[d][c][i] = (1.0*(n_sp_dc[i][d][c] + beta_w))/(ndc_sp_count[d][c] + spVocabSize*beta_w);
        }
    }
    createOutputFiles(Pd, Pc, P_lrp_given_c, P_rrp_given_c, P_v_given_d, P_w_given_dc, P_sp_given_dc);
}


int main(int argc, char const *argv[])
{
    if(argc < 2)
      {
        cout << "Incorrect number of arguments" << endl;
        return 0;
      }
    clock_t t;
    t = clock();
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
    cout << "Total running time = "<< (float)(clock() - t)/CLOCKS_PER_SEC << endl;
    outputResult();
    return 0;
}
