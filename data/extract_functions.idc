//usage: dans IDA, lancer le script. Le fichier de log est 
//dans le répertoire de l'exe avec l'extension .log

#include <idc.idc>

static is_executable(ea) {
    return (GetSegmentAttr(ea, SEGATTR_PERM) & SEGPERM_EXEC) != 0;
}

static get_output_file_name(){
    auto filename = GetInputFile();
    auto dot_pos = strstr(filename, ".");
    if (dot_pos) {
        filename = substr(filename, 0, dot_pos) + ".log";
    } else {
        filename = filename + ".txt";
    }
    
    auto file = fopen(filename, "w"); // Open file for writing
    if (!file) {
        Message("Failed to open file\n");
        return 0;
    }
    return file;
}

static main() {
    Message("hello\n");
    auto log_file = get_output_file_name();

    auto ea = FirstSeg(); // Start from the first segment
    auto seg = FirstSeg();
    auto p = 1;
    fprintf(log_file, "<head>\n");

    Message("read heads...");
    while (ea != BADADDR) {
        if (is_executable(ea)){
            fprintf(log_file, "%02X ", ea);
        }
        ea=next_head(ea, BADADDR) ;
    }
    Message("heads done\n");
    fprintf(log_file, "\n</head>\n");

    ea = FirstSeg();
    while (seg != BADADDR) {
        if (is_executable(ea)){
            fprintf(log_file, "<segment addr=%02X>\n", seg);
            while (ea != SegEnd(seg)){
                auto data =  Byte(ea);
                fprintf(log_file, "%02X ", data);
                ea = ea + 1;
            }
            fprintf(log_file, "\n</segment addr=%02X>\n", SegEnd(seg));
        }
        seg = NextSeg(seg);
        ea = seg;
    }
    auto func = NextFunction(FirstSeg());
    fprintf(log_file, "<functions addr=%02X>\n", ea);
    while (func != BADADDR){
        fprintf(log_file, "%02X ", func);
        func = NextFunction(func);
    }
    fprintf(log_file, "\n</functions>\n");
    fclose(log_file);    
    Message("done");
}
