#include "semafor.h"

Semafor::Semafor(int _count): licznikZasobow(_count) { }

int Semafor::wartosc() const
{
    return licznikZasobow;
}

void Semafor::sygnalizuj()
{
    //licznik na wylacznosc
    unique_lock<mutex> blokadaLicznika(mutexLicznika);
    //zwiekszamy licznik
    ++licznikZasobow;
    //powiadamiamy jeden oczekujacy watek, ze licznik
    //zostal zwiekszony
    warunekLicznikRoznyOdZera.notify_one();
}

void Semafor::czekaj()
{
    //licznik na wylacznosc
    unique_lock<mutex> blokadaLicznika(mutexLicznika);
    //jezeli licznik jest rowny 0 to nie mozemy go
    //zmniejszyc, czekamy az zostanie zwiekszony przez
    //inny watek
    while (licznikZasobow == 0)
    warunekLicznikRoznyOdZera.wait(blokadaLicznika);
    //zmniejszamy licznik
    --licznikZasobow;
}

bool Semafor::probuj_czekac()
{
     //licznik na wylacznosc
    unique_lock<mutex> blokadaLicznika(mutexLicznika);
    //jezeli licznik jest wiekszy od 0 to go zmniejszamy
    if (licznikZasobow != 0)
    {
        --licznikZasobow;
        return true;
    }
    //w przeciwnym razie informujemy ze nie udalo sie
    else
    return false;
}
