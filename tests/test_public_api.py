import unittest

from yapzeta_verifier import (
    CON, INS, PEND, SUP, SUPD, UNV, USER_LABEL, user_label, verify_claims
)


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        from yapzeta_verifier import checks
        checks.USE_NLI = False
        checks.USE_LLM = False
        self.chunk = {
            "id": "synthetic-1",
            "kaynak": "Synthetic source, stoa parameters",
            "metin": "Stoa kat sayısı: 2.",
            "etiket": "Saha kaydı",
            "meta": {"source_type": "primary_excavation", "source_id": "synthetic-source", "approval_status": "reported"},
        }

    def test_six_states_and_display_mapping(self):
        self.assertEqual({SUP, CON, INS, PEND, SUPD, UNV}, set(USER_LABEL))
        self.assertEqual(user_label(SUP), "DESTEKLENİYOR")
        self.assertEqual(user_label(CON), "ÇELİŞİYOR")
        self.assertEqual(user_label(SUPD), "ÇELİŞİYOR")
        self.assertEqual(user_label(INS), "YETERSİZ_KANIT")
        self.assertEqual(user_label(PEND), "YETERSİZ_KANIT")
        self.assertEqual(user_label(UNV), "YETERSİZ_KANIT")

    def test_synthetic_support_and_contradiction(self):
        rows = verify_claims(["Stoa iki katlıdır.", "Stoa üç katlıdır."], [self.chunk])
        self.assertEqual([r["verification_status"] for r in rows], [SUP, CON])
        self.assertTrue(rows[0]["evidence"][0]["quote_span"] in self.chunk["metin"])

    def test_missing_evidence_is_unverified(self):
        row = verify_claims(["Stoa iki katlıdır."], [])[0]
        self.assertEqual(row["verification_status"], UNV)

    def test_insufficient_and_pending_conflict_paths(self):
        from yapzeta_verifier import gate
        chat_note = {
            "id": "synthetic-chat", "kaynak": "Synthetic note",
            "metin": "Stoa kat sayısı: 2.", "etiket": "synthetic",
            "meta": {"source_type": "chat_note", "source_id": "synthetic-chat"},
        }
        pending = {
            "id": "synthetic-pending", "kaynak": "Synthetic field record",
            "metin": "Stoa kat sayısı: 2.", "etiket": "açık soru", "celiski": True,
            "meta": {"source_type": "primary_excavation", "source_id": "synthetic-pending", "approval_status": "reported"},
        }
        self.assertEqual(gate.evaluate("Stoa iki katlıdır.", [chat_note], {"synthetic-chat"})["verification_status"], INS)
        self.assertEqual(gate.evaluate("Stoa iki katlıdır.", [pending], {"synthetic-pending"})["verification_status"], PEND)

    def test_superseded_end_to_end(self):
        chunk = dict(self.chunk, metin="Eski kayıt: Stoa üç katlıdır. Güncel karar: Stoa iki katlıdır.")
        row = verify_claims(["Stoa üç katlıdır."], [chunk])[0]
        self.assertEqual(row["verification_status"], SUPD)
        self.assertEqual(row["user_label"], user_label(SUPD))

    def test_no_candidate_display_fields(self):
        row = verify_claims(["Stoa iki katlıdır."], [])[0]
        self.assertEqual(row["user_label"], user_label(UNV))
        self.assertTrue(row["rozet"])

    def test_invalid_evidence_display_fields(self):
        from yapzeta_verifier import gate
        for evidence in ([], [{"chunk_id": "missing", "quote_span": "x"}],
                         [{"chunk_id": self.chunk["id"], "quote_span": "not present"}]):
            row = gate.validate_evidence("Stoa iki katlıdır.", evidence, {self.chunk["id"]: self.chunk})
            self.assertEqual(row["verification_status"], UNV)
            self.assertEqual(row["user_label"], user_label(UNV))
            self.assertTrue(row["rozet"])


if __name__ == "__main__":
    unittest.main()
