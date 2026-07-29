"""
Load the 7 snapshot CSVs, de-duplicate multi-exchange listings, join them into
one security-level table, and normalize every column the model needs.

Public entry point:  build_security_table() -> pandas.DataFrame  (one row per security).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C


def _read(path: str) -> pd.DataFrame:
    """Read a snapshot CSV as strings and collapse multi-exchange duplicate rows.

    A security listed on several exchanges appears multiple times under the same
    `Símbolo`.  An *issuance* is one security, so we keep the first listing.
    """
    df = pd.read_csv(path, dtype=str)
    df = df.drop_duplicates(subset=["Símbolo"], keep="first")
    return df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_security_table() -> pd.DataFrame:
    """Return one clean row per security with normalized, model-ready columns."""
    p = C.csv_paths()
    issue = _read(p["issue"])
    amount = _read(p["amount"])
    struct = _read(p["struct"])
    coupon = _read(p["coupon"])
    issuer = _read(p["issuer"])
    rating = _read(p["rating"])
    master = _read(p["master"])

    # ---- master carries ISIN / exchange / issuer type; join the rest onto it ----
    keep_issue = ["Símbolo", "Descrição", "Data de emissão", "Data de vencimento",
                  "Prazo/Vencimento", "Tipo de duração (prazo)", "Tipo de colocação",
                  "Alocação de emissão", "Status da emissão", "Preço da oferta %"]
    keep_amount = ["Símbolo", "Valor da emissão inicial", "Valor da emissão inicial - Moeda",
                   "Valor pendente", "Valor de face"]
    keep_struct = ["Símbolo", "Tipo de resgate (call/put)", "Opção de resgate antecipado",
                   "Sinking fund", "Opção de conversão", "Classificação de senioridade",
                   "Tipo de vencimento", "Próxima data de call"]
    keep_coupon = ["Símbolo", "Cupom atual %", "Tipo de cupom", "Tipo de cupom atual",
                   "Frequência do cupom", "Base de contagem de dias do cupom"]
    keep_issuer = ["Símbolo", "Emissor", "País ou região do emissor", "Setor", "Indústria",
                   "Classificação S&P do emissor (longo prazo)",
                   "Classificação da Fitch do emissor (longo prazo)"]
    keep_rating = ["Símbolo", "Classificação S&P", "Classificação da Fitch"]

    df = (master[["Símbolo", "Descrição", "ISIN", "Fonte", "Tipo de emissor"]]
          .merge(issue[keep_issue], on="Símbolo", how="left", suffixes=("", "_i"))
          .merge(amount[keep_amount], on="Símbolo", how="left")
          .merge(struct[keep_struct], on="Símbolo", how="left")
          .merge(coupon[keep_coupon], on="Símbolo", how="left")
          .merge(issuer[keep_issuer], on="Símbolo", how="left")
          .merge(rating[keep_rating], on="Símbolo", how="left"))

    out = pd.DataFrame(index=df.index)

    # ---- identity ----
    out["symbol"] = df["Símbolo"]
    out["isin"] = df["ISIN"]
    out["description"] = df["Descrição"]
    out["issuer"] = df["Emissor"].fillna("(unknown)")
    out["country"] = df["País ou região do emissor"].fillna("(unknown)")
    out["sector"] = df["Setor"].fillna("(unknown)")
    out["industry"] = df["Indústria"]
    out["issuer_type"] = df["Tipo de emissor"].fillna("(unknown)")

    # ---- dates / tenor ----
    out["issue_date"] = pd.to_datetime(df["Data de emissão"], errors="coerce")
    out["maturity_date"] = pd.to_datetime(df["Data de vencimento"], errors="coerce")
    out["tenor_days"] = (out["maturity_date"] - out["issue_date"]).dt.days
    out["duration_type"] = df["Tipo de duração (prazo)"].map(C.DURATION_TYPE)

    # ---- coupon ----
    out["coupon_rate"] = _num(df["Cupom atual %"])
    out["coupon_type"] = df["Tipo de cupom"].map(C.COUPON_TYPE)
    out["coupon_freq"] = df["Frequência do cupom"]

    # ---- structure / optionality ----
    rtype = df["Tipo de resgate (call/put)"]
    out["callable"] = rtype.isin(C.REDEMPTION_CALLABLE).astype("Int64")
    out["puttable"] = rtype.isin(C.REDEMPTION_PUTTABLE).astype("Int64")
    out["sinkable"] = df["Sinking fund"].map(C.SINKABLE).astype("Int64")
    out["convertible"] = df["Opção de conversão"].map(C.CONVERTIBLE).astype("Int64")
    out["early_redeem"] = df["Opção de resgate antecipado"].map(C.EARLY_REDEEM).astype("Int64")
    out["seniority_ord"] = df["Classificação de senioridade"].map(C.SENIORITY_ORD)

    # ---- placement / allocation / status ----
    out["placement"] = df["Tipo de colocação"].map(C.PLACEMENT)
    out["allocation"] = df["Alocação de emissão"].map(C.ALLOCATION)
    out["defaulted"] = (df["Status da emissão"] == "Inadimplente").astype("Int64")

    # ---- size / currency ----
    out["amount_issued"] = _num(df["Valor da emissão inicial"])
    out["amount_ccy"] = df["Valor da emissão inicial - Moeda"]
    out["amount_outstanding"] = _num(df["Valor pendente"])
    out["log_amount"] = np.log1p(out["amount_issued"].clip(lower=0))

    # FX flag: issued in a currency other than the issuer country's local ccy
    local = out["country"].map(C.LOCAL_CCY_BY_COUNTRY)
    out["fx_flag"] = np.where(
        out["amount_ccy"].notna() & local.notna(),
        (out["amount_ccy"] != local).astype(float),
        np.nan,
    )

    # ---- ratings (issuer LT: S&P, fallback Fitch) ----
    sp = df["Classificação S&P do emissor (longo prazo)"].map(C.rating_to_notch)
    fitch = df["Classificação da Fitch do emissor (longo prazo)"].map(C.rating_to_notch)
    out["issuer_rating_notch"] = sp.fillna(fitch)
    out["has_rating"] = out["issuer_rating_notch"].notna().astype(int)

    return out


if __name__ == "__main__":
    t = build_security_table()
    pd.set_option("display.width", 160, "display.max_columns", 40)
    print("securities:", len(t))
    print("issuers:", t["issuer"].nunique(), " countries:", t["country"].nunique())
    print("issue-date coverage:", t["issue_date"].notna().mean().round(4))
    print(t[["symbol", "issuer", "country", "issue_date", "coupon_rate", "coupon_type",
             "callable", "tenor_days", "issuer_rating_notch"]].head(8).to_string())
    print("\ncoupon_type:\n", t["coupon_type"].value_counts(dropna=False).to_string())
    print("\ncallable rate:", t["callable"].mean().round(3),
          " fx rate:", np.nanmean(t["fx_flag"].astype(float)).round(3))
