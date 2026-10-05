/* Valutazione nel browser del modello finale esportato da src/tesi_bev/esporta_web.py.
 *
 * Riproduce la pipeline Python: codifica_ml -> StandardScaler -> LightGBM (somma delle foglie + sigmoide).
 * I contributi delle variabili sono i valori SHAP esatti (algoritmo TreeSHAP di Lundberg et al., 2018),
 * gli stessi di shap.TreeExplainer / booster.predict(pred_contrib=True).
 * Funziona sia nella pagina (window.ModelloBEV) sia in Node (require) per i test.
 */
(function (radice) {
  "use strict";

  /* Profilo con le colonne del dataset originale -> vettore delle 18 feature (come dati.codifica_ml). */
  function codifica(modello, profilo) {
    return modello.codifica.map(function (r) {
      var v = profilo[r.sorgente];
      if (v === undefined || v === null || v === "") throw new Error("Valore mancante: " + r.sorgente);
      var x = r.tipo === "mappa" ? r.mappa[v] : r.tipo === "dummy" ? (v === r.valore ? 1 : 0) : Number(v);
      if (typeof x !== "number" || !isFinite(x)) throw new Error("Valore non riconosciuto per " + r.sorgente + ": " + v);
      return x;
    });
  }

  function standardizza(modello, x) {
    return x.map(function (v, i) { return (v - modello.media[i]) / modello.scala[i]; });
  }

  /* Albero: [feature, soglia, sinistro, destro, valore, copertura]; nelle foglie feature = -1. */
  function foglia(a, x) {
    var n = 0;
    while (a[0][n] >= 0) n = x[a[0][n]] <= a[1][n] ? a[2][n] : a[3][n];
    return n;
  }

  function punteggio(modello, xs) {
    var s = 0;
    for (var t = 0; t < modello.alberi.length; t++) {
      var a = modello.alberi[t];
      s += a[4][foglia(a, xs)];
    }
    return s;
  }

  /* --- TreeSHAP (stessa struttura di tree_shap in shap/cext/tree_shap.h) --- */

  function estendi(p, d, z, o, i) {
    p[d] = { i: i, z: z, o: o, w: d === 0 ? 1 : 0 };
    for (var k = d - 1; k >= 0; k--) {
      p[k + 1].w += o * p[k].w * (k + 1) / (d + 1);
      p[k].w = z * p[k].w * (d - k) / (d + 1);
    }
  }

  function riavvolgi(p, d, j) {
    var o = p[j].o, z = p[j].z, succ = p[d].w, k;
    for (k = d - 1; k >= 0; k--) {
      if (o !== 0) {
        var tmp = p[k].w;
        p[k].w = succ * (d + 1) / ((k + 1) * o);
        succ = tmp - p[k].w * z * (d - k) / (d + 1);
      } else {
        p[k].w = p[k].w * (d + 1) / (z * (d - k));
      }
    }
    for (k = j; k < d; k++) { p[k].i = p[k + 1].i; p[k].z = p[k + 1].z; p[k].o = p[k + 1].o; }
  }

  function sommaRiavvolta(p, d, j) {
    var o = p[j].o, z = p[j].z, succ = p[d].w, tot = 0;
    for (var k = d - 1; k >= 0; k--) {
      if (o !== 0) {
        var tmp = succ * (d + 1) / ((k + 1) * o);
        tot += tmp;
        succ = p[k].w - tmp * z * (d - k) / (d + 1);
      } else if (z !== 0) {
        tot += p[k].w / z / ((d - k) / (d + 1));
      }
    }
    return tot;
  }

  function shapAlbero(a, x, phi) {
    var f = a[0], s = a[1], sx = a[2], dx = a[3], v = a[4], c = a[5];
    function ricorri(n, padre, d, z, o, i) {
      var p = padre.slice(0, d).map(function (e) { return { i: e.i, z: e.z, o: e.o, w: e.w }; });
      estendi(p, d, z, o, i);
      if (f[n] < 0) {
        for (var k = 1; k <= d; k++) phi[p[k].i] += sommaRiavvolta(p, d, k) * (p[k].o - p[k].z) * v[n];
        return;
      }
      var caldo = x[f[n]] <= s[n] ? sx[n] : dx[n], freddo = caldo === sx[n] ? dx[n] : sx[n];
      var iz = 1, io = 1, j = 0;
      while (j <= d && p[j].i !== f[n]) j++;
      if (j <= d) { iz = p[j].z; io = p[j].o; riavvolgi(p, d, j); d--; }
      ricorri(caldo, p, d + 1, iz * c[caldo] / c[n], io, f[n]);
      ricorri(freddo, p, d + 1, iz * c[freddo] / c[n], 0, f[n]);
    }
    ricorri(0, [], 0, 1, 1, -1);
  }

  function contributi(modello, xs) {
    var phi = modello.colonne.map(function () { return 0; });
    for (var t = 0; t < modello.alberi.length; t++) shapAlbero(modello.alberi[t], xs, phi);
    return phi;
  }

  /* Valore atteso del punteggio (media pesata delle foglie): base + somma dei contributi = punteggio. */
  function valoreAtteso(modello) {
    var tot = 0;
    modello.alberi.forEach(function (a) {
      function media(n) { return a[0][n] < 0 ? a[4][n] : (a[5][a[2][n]] * media(a[2][n]) + a[5][a[3][n]] * media(a[3][n])) / a[5][n]; }
      tot += media(0);
    });
    return tot;
  }

  /* Profilo -> { probabilita, punteggio (log-odds), contributi SHAP in log-odds per feature }. */
  function predici(modello, profilo, conShap) {
    var xs = standardizza(modello, codifica(modello, profilo));
    var p = punteggio(modello, xs);
    return { probabilita: 1 / (1 + Math.exp(-p)), punteggio: p, contributi: conShap === false ? null : contributi(modello, xs) };
  }

  var api = { valoreAtteso: valoreAtteso, codifica: codifica, standardizza: standardizza, punteggio: punteggio, contributi: contributi, predici: predici };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else radice.ModelloBEV = api;
})(typeof window !== "undefined" ? window : this);
