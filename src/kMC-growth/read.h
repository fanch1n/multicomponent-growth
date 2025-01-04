#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm> // for std::copy
bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs);
std::vector<int> getData(std::string fileName);
std::vector<string> splitString(std::string s);
/*----------------------------------------*/
bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(str);
    }
    //Close The File
    in.close();
    return true;
}
/*----------------------------------------*/
std::vector<int> getData(std::string fileName)
{
    cerr << "start loading file: " << fileName << endl;
    std::vector<int> data;
    std::ifstream infile(fileName, std::ios::in);
    // Check if object is valid
    if(!infile.is_open())
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        exit(1);
    }
    int num = 0;
    std::string line;
    while (std::getline(infile, line))
    {   
        if(line[0] == '#'){
            continue;
        }
        else{
            std::vector<string> entries = splitString(line);
            for(int i = 0; i < entries.size(); i++){
                num = int(std::stof(entries[i]));
                data.push_back(num);
            }
        }
    }
    infile.close();
    return data;
}
/*----------------------------------------*/
//std::vector<int> getData(std::string fileName)
//{
//    std::vector<int> data;
//    // Open the File
//    std::ifstream infile(fileName, std::ios::in);
//    // Check if object is valid
//    if(!infile.is_open())
//    {
//        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
//        exit(1);
//    }
//
//    double num = 0.0;
//    //keep storing values from the text file as long as the data exists;
//    while (infile >> num )
//    {   
//        data.push_back(int(num));
//    }
//    cerr << "print read file" << endl;
//    for(int i = 0; i < data.size(); i ++) cerr << data[i] << endl;
//   
//    //Close The File
//    infile.close();
//    return data;
//}
//
std::vector<string> splitString(string s){
    vector<string> v;
    string temp = "";
    int i = 0;
    while(i < s.length() and s[i] == ' ') i ++;
    if( i < s.length()){
        while(i < s.length()){
            if(s[i] != ' '){
                temp.push_back(s[i]);
                i ++;
            }
            else{
               v.push_back(temp);
               temp = "";
               while(i < s.length() and s[i] == ' ') i ++;
            }
       }
    }
    if(temp != "") v.push_back(temp);
    return v;
}
